#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-t2av}
VARIANT=${2:-dev}
MODEL_ROOT=${LTX25_MODEL_ROOT:-models/Lightricks/LTX-2.5}
DATA_ROOT=${LTX25_DATA_ROOT:-data/diffsynth_example_dataset/ltx2/LTX-2.5-${MODE}}
METADATA=${LTX25_METADATA:-${DATA_ROOT}/metadata.json}
NUM_EPOCHS=${LTX25_NUM_EPOCHS:-5}
DATASET_REPEAT=${LTX25_DATASET_REPEAT:-100}
MODE_CONFIG="examples/ltx2/model_training/ltx25/${MODE}.yaml"

case "$MODE" in
    t2av|i2av|video_extend_prefix|video_extend_suffix|a2v|v2a)
        DATA_FILE_KEYS="video,input_audio"
        ;;
    v2v_ic_lora)
        DATA_FILE_KEYS="video,reference_video"
        ;;
    video_inpainting|video_outpainting)
        DATA_FILE_KEYS="video,video_mask"
        ;;
    t2a|audio_extend_prefix|audio_extend_suffix)
        DATA_FILE_KEYS="input_audio"
        ;;
    audio_inpainting)
        DATA_FILE_KEYS="input_audio,audio_mask"
        ;;
    a2a_ic_lora)
        DATA_FILE_KEYS="input_audio,reference_audio"
        ;;
    av2av_ic_lora)
        DATA_FILE_KEYS="video,input_audio,reference_video,reference_audio"
        ;;
    *)
        echo "Unsupported LTX-2.5 training mode: $MODE" >&2
        exit 1
        ;;
esac

accelerate launch examples/ltx2/model_training/train_ltx25.py \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_metadata_path "$METADATA" \
  --data_file_keys "$DATA_FILE_KEYS" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --ltx25_model_root "$MODEL_ROOT" \
  --ltx25_variant "$VARIANT" \
  --training_mode_config "$MODE_CONFIG" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.5-${MODE}-${VARIANT}-full-cache" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

accelerate launch \
  --config_file examples/ltx2/model_training/full/accelerate_config_ltx2_5_deepspeed_zero3.yaml \
  examples/ltx2/model_training/train_ltx25.py \
  --dataset_base_path "./models/train/LTX2.5-${MODE}-${VARIANT}-full-cache" \
  --data_file_keys "$DATA_FILE_KEYS" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --dataset_repeat "$DATASET_REPEAT" \
  --ltx25_model_root "$MODEL_ROOT" \
  --ltx25_variant "$VARIANT" \
  --training_mode_config "$MODE_CONFIG" \
  --learning_rate 1e-5 \
  --num_epochs "$NUM_EPOCHS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.5-${MODE}-${VARIANT}-full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --initialize_model_on_cpu \
  --enable_tensorboard_log \
  --task "sft:train"
