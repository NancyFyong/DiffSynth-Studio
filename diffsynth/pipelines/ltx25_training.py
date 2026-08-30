from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..diffusion.base_pipeline import PipelineUnit
from ..models.ltx2_common import AudioLatentShape, VideoLatentShape, VIDEO_SCALE_FACTORS, get_pixel_coords
from ..utils.data.audio import convert_to_stereo
from .ltx25_audio_video import LTX25AudioVideoPipeline, LTX25AudioVideoUnit_PromptEmbedder


@dataclass
class LTX25TrainingCondition:
    type: str
    probability: float = 1.0
    temporal_boundary: int | None = None
    spatial_region: list[int] | None = None
    downscale_factor: int = 1
    temporal_scale_factor: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LTX25TrainingCondition":
        return cls(
            type=data["type"],
            probability=float(data.get("probability", 1.0)),
            temporal_boundary=data.get("temporal_boundary"),
            spatial_region=data.get("spatial_region"),
            downscale_factor=int(data.get("downscale_factor", 1)),
            temporal_scale_factor=int(data.get("temporal_scale_factor", 1)),
        )


@dataclass
class LTX25TrainingModalityConfig:
    is_generated: bool
    conditions: list[LTX25TrainingCondition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LTX25TrainingModalityConfig":
        return cls(
            is_generated=bool(data["is_generated"]),
            conditions=[LTX25TrainingCondition.from_dict(condition) for condition in data.get("conditions", [])],
        )


@dataclass
class LTX25FlexibleTrainingConfig:
    video: LTX25TrainingModalityConfig | None = None
    audio: LTX25TrainingModalityConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LTX25FlexibleTrainingConfig":
        config = cls(
            video=LTX25TrainingModalityConfig.from_dict(data["video"]) if data.get("video") is not None else None,
            audio=LTX25TrainingModalityConfig.from_dict(data["audio"]) if data.get("audio") is not None else None,
        )
        config.validate()
        return config

    def validate(self):
        if self.video is None and self.audio is None:
            raise ValueError("At least one modality must be configured.")
        if not any(modality is not None and modality.is_generated for modality in (self.video, self.audio)):
            raise ValueError("At least one modality must have is_generated=True.")
        self._validate_modality(
            "video",
            self.video,
            {"first_frame", "prefix", "suffix", "mask", "spatial_crop", "reference"},
        )
        self._validate_modality("audio", self.audio, {"prefix", "suffix", "mask", "reference"})

    @staticmethod
    def _validate_modality(name, modality, supported_conditions):
        if modality is None:
            return
        reference_count = 0
        for condition in modality.conditions:
            if condition.type not in supported_conditions:
                raise ValueError(f"Unsupported {name} condition: {condition.type}")
            if not 0.0 <= condition.probability <= 1.0:
                raise ValueError(f"{name} condition probability must be in [0, 1].")
            if condition.type in {"prefix", "suffix"} and (
                condition.temporal_boundary is None or condition.temporal_boundary <= 0
            ):
                raise ValueError(f"{name} {condition.type} requires a positive temporal_boundary.")
            if condition.type == "spatial_crop":
                if condition.spatial_region is None or len(condition.spatial_region) != 4:
                    raise ValueError("video spatial_crop requires [y1, x1, y2, x2].")
            if condition.type == "reference":
                reference_count += 1
                if condition.downscale_factor <= 0 or condition.temporal_scale_factor <= 0:
                    raise ValueError(f"{name} reference scale factors must be positive.")
        if reference_count > 1:
            raise ValueError(f"Only one {name} reference condition is supported.")


def _active(condition: LTX25TrainingCondition, device: torch.device) -> bool:
    return condition.probability >= 1.0 or bool(torch.rand((), device=device) < condition.probability)


def _video_positions(pipe, latents, frame_rate, spatial_scale_factor=1, temporal_scale_factor=1):
    latent_shape = VideoLatentShape.from_torch_shape(latents.shape)
    latent_coords = pipe.video_patchifier.get_patch_grid_bounds(output_shape=latent_shape, device=pipe.device)
    positions = get_pixel_coords(latent_coords, VIDEO_SCALE_FACTORS, True).float()
    positions[:, 0] = positions[:, 0] * temporal_scale_factor / frame_rate
    positions[:, 1] *= spatial_scale_factor
    positions[:, 2] *= spatial_scale_factor
    return positions.to(pipe.torch_dtype)


def _audio_positions(pipe, latents, temporal_scale_factor=1):
    latent_shape = AudioLatentShape.from_torch_shape(latents.shape)
    positions = pipe.audio_patchifier.get_patch_grid_bounds(latent_shape, device=pipe.device).to(pipe.torch_dtype)
    positions[:, 0] *= temporal_scale_factor
    return positions


class LTX25FlexibleTrainingEncoder(PipelineUnit):
    def __init__(self, config: LTX25FlexibleTrainingConfig):
        onload_model_names = []
        if config.video is not None:
            onload_model_names.append("video_vae_encoder")
        if config.audio is not None:
            onload_model_names.append("audio_vae_encoder")
        super().__init__(
            input_params=(
                "input_video",
                "input_audio",
                "reference_video",
                "reference_audio",
                "video_mask",
                "audio_mask",
                "height",
                "width",
                "num_frames",
                "frame_rate",
                "tiled",
                "tile_size_in_pixels",
                "tile_overlap_in_pixels",
            ),
            output_params=(
                "video_input_latents",
                "audio_input_latents",
                "video_positions",
                "audio_positions",
                "video_denoise_mask",
                "audio_denoise_mask",
                "video_reference_latents",
                "video_reference_positions",
                "audio_reference_latents",
                "audio_reference_positions",
            ),
            onload_model_names=tuple(onload_model_names),
        )
        self.config = config

    @staticmethod
    def _video_frames(video):
        if video is None:
            return None
        if isinstance(video, (list, tuple)) and len(video) > 0 and isinstance(video[0], (list, tuple)):
            if len(video) != 1:
                raise ValueError("LTX-2.5 training accepts one reference video per sample.")
            return video[0]
        return video

    def _encode_video(self, pipe, video, tiled, tile_size_in_pixels, tile_overlap_in_pixels):
        video = self._video_frames(video)
        if video is None:
            raise ValueError("The selected training mode requires video data.")
        video = pipe.preprocess_video(video)
        return pipe.video_vae_encoder.encode(video, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )

    @staticmethod
    def _audio_waveform(audio):
        if audio is None:
            raise ValueError("The selected training mode requires audio data.")
        waveform, sample_rate = audio
        return convert_to_stereo(waveform), sample_rate

    def _encode_audio(self, pipe, audio):
        waveform, sample_rate = self._audio_waveform(audio)
        mel = pipe.audio_processor.waveform_to_mel(
            waveform.unsqueeze(0),
            waveform_sample_rate=sample_rate,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        return pipe.audio_vae_encoder(mel).to(dtype=pipe.torch_dtype, device=pipe.device)

    def _video_mask(self, video_mask, target):
        frames = self._video_frames(video_mask)
        if frames is None:
            raise ValueError("video mask conditioning requires video_mask data.")
        values = []
        for frame in frames:
            if torch.is_tensor(frame):
                value = frame.float()
                if value.ndim == 3:
                    value = value.mean(dim=0)
            else:
                value = torch.from_numpy(np.asarray(frame.convert("L"), dtype=np.float32) / 255.0)
            values.append(value)
        mask = torch.stack(values).unsqueeze(0).unsqueeze(0).to(device=target.device)
        mask = F.interpolate(mask, size=target.shape[2:], mode="trilinear", align_corners=False)
        return mask >= 0.5

    def _audio_mask(self, audio_mask, target):
        waveform, _ = self._audio_waveform(audio_mask)
        mask = waveform.abs().mean(dim=0).reshape(1, 1, -1).to(target.device)
        mask = F.interpolate(mask, size=target.shape[2], mode="linear", align_corners=False)
        return (mask >= 0.5).unsqueeze(-1)

    def _video_conditioning_mask(self, target, video_mask):
        mask = torch.zeros((target.shape[0], 1, *target.shape[2:]), dtype=torch.bool, device=target.device)
        for condition in self.config.video.conditions:
            if not _active(condition, target.device) or condition.type == "reference":
                continue
            if condition.type == "first_frame":
                mask[:, :, :1] = True
            elif condition.type == "prefix":
                mask[:, :, :min(condition.temporal_boundary, target.shape[2])] = True
            elif condition.type == "suffix":
                mask[:, :, max(target.shape[2] - condition.temporal_boundary, 0):] = True
            elif condition.type == "mask":
                mask |= self._video_mask(video_mask, target)
            elif condition.type == "spatial_crop":
                y1, x1, y2, x2 = condition.spatial_region
                height, width = target.shape[-2:]
                y1, y2 = max(y1 // 32, 0), min(-(-y2 // 32), height)
                x1, x2 = max(x1 // 32, 0), min(-(-x2 // 32), width)
                mask[:, :, :, y1:y2, x1:x2] = True
        return mask

    def _audio_conditioning_mask(self, target, audio_mask):
        mask = torch.zeros((target.shape[0], 1, target.shape[2], 1), dtype=torch.bool, device=target.device)
        for condition in self.config.audio.conditions:
            if not _active(condition, target.device) or condition.type == "reference":
                continue
            if condition.type == "prefix":
                mask[:, :, :min(condition.temporal_boundary, target.shape[2])] = True
            elif condition.type == "suffix":
                mask[:, :, max(target.shape[2] - condition.temporal_boundary, 0):] = True
            elif condition.type == "mask":
                mask |= self._audio_mask(audio_mask, target)
        return mask

    @staticmethod
    def _reference_condition(config, device):
        for condition in config.conditions:
            if condition.type == "reference" and _active(condition, device):
                return condition
        return None

    def process(
        self,
        pipe,
        input_video,
        input_audio,
        reference_video,
        reference_audio,
        video_mask,
        audio_mask,
        height,
        width,
        num_frames,
        frame_rate,
        tiled,
        tile_size_in_pixels,
        tile_overlap_in_pixels,
    ):
        pipe.load_models_to_device(self.onload_model_names)
        outputs = {
            "video_input_latents": None,
            "audio_input_latents": None,
            "video_positions": None,
            "audio_positions": None,
            "video_denoise_mask": None,
            "audio_denoise_mask": None,
            "video_reference_latents": None,
            "video_reference_positions": None,
            "audio_reference_latents": None,
            "audio_reference_positions": None,
        }
        if self.config.video is not None:
            video_target = self._encode_video(pipe, input_video, tiled, tile_size_in_pixels, tile_overlap_in_pixels)
            video_conditioning = self._video_conditioning_mask(video_target, video_mask)
            if not self.config.video.is_generated:
                video_conditioning[:] = True
            outputs["video_input_latents"] = video_target
            outputs["video_positions"] = _video_positions(pipe, video_target, frame_rate)
            outputs["video_denoise_mask"] = (~video_conditioning).to(video_target.dtype)
            condition = self._reference_condition(self.config.video, video_target.device)
            if condition is not None:
                reference = self._encode_video(
                    pipe,
                    reference_video,
                    tiled,
                    tile_size_in_pixels,
                    tile_overlap_in_pixels,
                )
                outputs["video_reference_latents"] = reference
                outputs["video_reference_positions"] = _video_positions(
                    pipe,
                    reference,
                    frame_rate,
                    condition.downscale_factor,
                    condition.temporal_scale_factor,
                )
        if self.config.audio is not None:
            audio_target = self._encode_audio(pipe, input_audio)
            audio_conditioning = self._audio_conditioning_mask(audio_target, audio_mask)
            if not self.config.audio.is_generated:
                audio_conditioning[:] = True
            outputs["audio_input_latents"] = audio_target
            outputs["audio_positions"] = _audio_positions(pipe, audio_target)
            outputs["audio_denoise_mask"] = (~audio_conditioning).to(audio_target.dtype)
            condition = self._reference_condition(self.config.audio, audio_target.device)
            if condition is not None:
                reference = self._encode_audio(pipe, reference_audio)
                outputs["audio_reference_latents"] = reference
                outputs["audio_reference_positions"] = _audio_positions(
                    pipe,
                    reference,
                    condition.temporal_scale_factor,
                )
        return outputs


def model_fn_ltx25_flexible(
    dit,
    video_latents=None,
    video_input_latents=None,
    video_denoise_mask=None,
    video_positions=None,
    video_context=None,
    video_reference_latents=None,
    video_reference_positions=None,
    audio_latents=None,
    audio_input_latents=None,
    audio_denoise_mask=None,
    audio_positions=None,
    audio_context=None,
    audio_reference_latents=None,
    audio_reference_positions=None,
    video_patchifier=None,
    audio_patchifier=None,
    timestep=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    timestep = timestep.float() / 1000.0
    video_shape = None
    audio_shape = None
    if video_latents is not None:
        _, _, frames, height, width = video_latents.shape
        video_shape = (frames, height, width)
        video_latents = video_patchifier.patchify(video_latents)
        video_sequence_length = video_latents.shape[1]
        video_timesteps = timestep.repeat(1, video_sequence_length, 1)
        if video_input_latents is not None:
            video_mask = video_patchifier.patchify(video_denoise_mask)
            video_input = video_patchifier.patchify(video_input_latents)
            video_latents = video_latents * video_mask + video_input * (1.0 - video_mask)
            video_timesteps = video_timesteps * video_mask
        if video_reference_latents is not None:
            video_reference = video_patchifier.patchify(video_reference_latents)
            video_latents = torch.cat([video_latents, video_reference], dim=1)
            video_positions = torch.cat([video_positions, video_reference_positions], dim=2)
            video_reference_timesteps = timestep.repeat(1, video_reference.shape[1], 1) * 0.0
            video_timesteps = torch.cat([video_timesteps, video_reference_timesteps], dim=1)
    else:
        video_sequence_length = None
        video_timesteps = None
    if audio_latents is not None:
        _, audio_channels, _, mel_bins = audio_latents.shape
        audio_shape = (audio_channels, mel_bins)
        audio_latents = audio_patchifier.patchify(audio_latents)
        audio_sequence_length = audio_latents.shape[1]
        audio_timesteps = timestep.repeat(1, audio_sequence_length, 1)
        if audio_input_latents is not None:
            audio_mask = audio_patchifier.patchify(audio_denoise_mask)
            audio_input = audio_patchifier.patchify(audio_input_latents)
            audio_latents = audio_latents * audio_mask + audio_input * (1.0 - audio_mask)
            audio_timesteps = audio_timesteps * audio_mask
        if audio_reference_latents is not None:
            audio_reference = audio_patchifier.patchify(audio_reference_latents)
            audio_latents = torch.cat([audio_latents, audio_reference], dim=1)
            audio_positions = torch.cat([audio_positions, audio_reference_positions], dim=2)
            audio_reference_timesteps = timestep.repeat(1, audio_reference.shape[1], 1) * 0.0
            audio_timesteps = torch.cat([audio_timesteps, audio_reference_timesteps], dim=1)
    else:
        audio_sequence_length = None
        audio_timesteps = None
    video_prediction, audio_prediction = dit(
        video_latents=video_latents,
        video_positions=video_positions,
        video_context=video_context,
        video_timesteps=video_timesteps,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_context=audio_context,
        audio_timesteps=audio_timesteps,
        sigma=timestep,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    if video_prediction is not None:
        video_prediction = video_patchifier.unpatchify_video(
            video_prediction[:, :video_sequence_length],
            *video_shape,
        )
    if audio_prediction is not None:
        audio_prediction = audio_patchifier.unpatchify_audio(
            audio_prediction[:, :audio_sequence_length],
            *audio_shape,
        )
    return video_prediction, audio_prediction


def _masked_flow_matching_loss(prediction, target, denoise_mask):
    loss = (prediction.float() - target.float()).square() * denoise_mask.float()
    denominator = denoise_mask.float().sum() * prediction.shape[1]
    return loss.sum() / denominator.clamp_min(1.0)


def flow_match_ltx25_flexible_loss(pipe, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    video_target = inputs.get("video_input_latents")
    audio_target = inputs.get("audio_input_latents")
    video_mask = inputs.get("video_denoise_mask")
    audio_mask = inputs.get("audio_denoise_mask")
    video_target_prediction = None
    audio_target_prediction = None
    if video_target is not None:
        if torch.any(video_mask):
            video_noise = torch.randn_like(video_target)
            inputs["video_latents"] = pipe.scheduler.add_noise(video_target, video_noise, timestep)
            video_target_prediction = pipe.scheduler.training_target(video_target, video_noise, timestep)
        else:
            inputs["video_latents"] = video_target
        inputs["video_input_latents"] = video_target
    if audio_target is not None:
        if torch.any(audio_mask):
            audio_noise = torch.randn_like(audio_target)
            inputs["audio_latents"] = pipe.scheduler.add_noise(audio_target, audio_noise, timestep)
            audio_target_prediction = pipe.scheduler.training_target(audio_target, audio_noise, timestep)
        else:
            inputs["audio_latents"] = audio_target
        inputs["audio_input_latents"] = audio_target
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    video_prediction, audio_prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = None
    if video_target_prediction is not None:
        loss = _masked_flow_matching_loss(video_prediction, video_target_prediction, video_mask)
    if audio_target_prediction is not None:
        audio_loss = _masked_flow_matching_loss(audio_prediction, audio_target_prediction, audio_mask)
        loss = audio_loss if loss is None else loss + audio_loss
    if loss is None:
        raise ValueError("The selected LTX-2.5 training sample has no generated tokens.")
    return loss * pipe.scheduler.training_weight(timestep)


class LTX25FlexibleTrainingPipeline(LTX25AudioVideoPipeline):
    def configure_training(self, config: LTX25FlexibleTrainingConfig):
        config.validate()
        self.training_config = config
        self.units = [
            LTX25AudioVideoUnit_PromptEmbedder(),
            LTX25FlexibleTrainingEncoder(config),
        ]
        self.model_fn = model_fn_ltx25_flexible
