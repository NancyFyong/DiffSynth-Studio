import argparse
from pathlib import Path

import accelerate
import torch
import yaml

from diffsynth.core import ModelConfig, UnifiedDataset
from diffsynth.core.data.operators import LoadPureAudioWithTorchaudio, RouteByType, SequencialProcess, ToAbsolutePath
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    ModelLogger,
    add_general_config,
    add_video_size_config,
    launch_data_process_task,
    launch_training_task,
)
from diffsynth.pipelines.ltx25_training import (
    LTX25FlexibleTrainingConfig,
    LTX25FlexibleTrainingPipeline,
    flow_match_ltx25_flexible_loss,
)


def load_training_config(path):
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("LTX-2.5 training mode config must be a mapping.")
    return LTX25FlexibleTrainingConfig.from_dict(data.get("training_strategy", data))


def build_ltx25_model_configs(model_root, variant, task):
    if variant not in {"dev", "distilled"}:
        raise ValueError("ltx25_variant must be dev or distilled.")
    root = Path(model_root)
    transformer = root / "diffusion_models" / f"ltx-2.5-22b-{variant}-transformer-bf16.safetensors"
    if task.endswith(":train"):
        return [ModelConfig(path=str(transformer))]
    gemma = root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    video_vae = root / "vae" / "ltx-2.5-video-vae-bf16.safetensors"
    audio_vae = root / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
    return [
        ModelConfig(path=str(gemma)),
        ModelConfig(path=[str(gemma), str(transformer)]),
        ModelConfig(path=str(video_vae)),
        ModelConfig(path=str(audio_vae)),
    ]


class LTX25TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        ltx25_model_root,
        ltx25_variant,
        training_mode_config,
        height,
        width,
        num_frames,
        frame_rate,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=None,
        task="sft",
        device="cpu",
    ):
        super().__init__()
        self.training_config = load_training_config(training_mode_config)
        model_configs = build_ltx25_model_configs(ltx25_model_root, ltx25_variant, task)
        gemma_path = Path(ltx25_model_root) / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
        self.pipe = LTX25FlexibleTrainingPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            gemma_path=gemma_path,
        )
        self.pipe.configure_training(self.training_config)
        self.pipe = self.split_pipeline_units(
            task,
            self.pipe,
            trainable_models,
            lora_base_model,
            remove_unnecessary_params=True,
            loss_required_params=(
                "video_input_latents",
                "audio_input_latents",
                "video_denoise_mask",
                "audio_denoise_mask",
                "video_positions",
                "audio_positions",
                "video_reference_latents",
                "video_reference_positions",
                "audio_reference_latents",
                "audio_reference_positions",
            ),
            force_remove_params_shared=("video_latents", "audio_latents"),
            force_remove_params_nega=("video_context", "audio_context"),
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: flow_match_ltx25_flexible_loss(
                pipe,
                **inputs_shared,
                **inputs_posi,
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: flow_match_ltx25_flexible_loss(
                pipe,
                **inputs_shared,
                **inputs_posi,
            ),
        }

    def get_pipeline_inputs(self, data):
        video = data.get("video")
        height = video[0].size[1] if video is not None else self.height
        width = video[0].size[0] if video is not None else self.width
        num_frames = len(video) if video is not None else self.num_frames
        inputs_shared = {
            "input_video": video,
            "input_audio": data.get("input_audio"),
            "reference_video": data.get("reference_video", data.get("in_context_videos")),
            "reference_audio": data.get("reference_audio"),
            "video_mask": data.get("video_mask"),
            "audio_mask": data.get("audio_mask"),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": data.get("frame_rate", self.frame_rate),
            "cfg_scale": 1.0,
            "tiled": False,
            "tile_size_in_pixels": 512,
            "tile_overlap_in_pixels": 128,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "video_patchifier": self.pipe.video_patchifier,
            "audio_patchifier": self.pipe.audio_patchifier,
        }
        return inputs_shared, {"prompt": data["prompt"]}, {}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)


def ltx25_parser():
    parser = argparse.ArgumentParser(description="LTX-2.5 flexible training.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--ltx25_model_root", type=str, default="models/Lightricks/LTX-2.5")
    parser.add_argument("--ltx25_variant", choices=("dev", "distilled"), default="dev")
    parser.add_argument("--training_mode_config", type=str, required=True)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--initialize_model_on_cpu", action="store_true")
    return parser


def build_dataset(args):
    video_processor = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=32,
        width_division_factor=32,
        num_frames=args.num_frames,
        time_division_factor=8,
        time_division_remainder=1,
        frame_rate=args.frame_rate,
        fix_frame_rate=True,
    )
    audio_operator = ToAbsolutePath(args.dataset_base_path) >> LoadPureAudioWithTorchaudio(
        max_audio_duration=args.num_frames / args.frame_rate,
        padding=True,
    )
    reference_video_operator = RouteByType(operator_map=[
        (str, video_processor),
        (list, SequencialProcess(video_processor)),
    ])
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            "input_audio": audio_operator,
            "reference_video": reference_video_operator,
            "in_context_videos": reference_video_operator,
            "reference_audio": audio_operator,
            "video_mask": video_processor,
            "audio_mask": audio_operator,
        },
    )


def main():
    args = ltx25_parser().parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = build_dataset(args)
    model = LTX25TrainingModule(
        ltx25_model_root=args.ltx25_model_root,
        ltx25_variant=args.ltx25_variant,
        training_mode_config=args.training_mode_config,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
