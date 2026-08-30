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
        LORA_TARGET_MODULES="to_k,to_q,to_v,to_out.0"
        ;;
    v2v_ic_lora)
        DATA_FILE_KEYS="video,reference_video"
        LORA_TARGET_MODULES="attn1.to_k,attn1.to_q,attn1.to_v,attn1.to_out.0,attn2.to_k,attn2.to_q,attn2.to_v,attn2.to_out.0,ff.net.0.proj,ff.net.2"
        ;;
    video_inpainting|video_outpainting)
        DATA_FILE_KEYS="video,video_mask"
        LORA_TARGET_MODULES="attn1.to_k,attn1.to_q,attn1.to_v,attn1.to_out.0,attn2.to_k,attn2.to_q,attn2.to_v,attn2.to_out.0,ff.net.0.proj,ff.net.2"
        ;;
    t2a|audio_extend_prefix|audio_extend_suffix)
        DATA_FILE_KEYS="input_audio"
        LORA_TARGET_MODULES="audio_attn1.to_k,audio_attn1.to_q,audio_attn1.to_v,audio_attn1.to_out.0,audio_attn2.to_k,audio_attn2.to_q,audio_attn2.to_v,audio_attn2.to_out.0,audio_ff.net.0.proj,audio_ff.net.2"
        ;;
    audio_inpainting)
        DATA_FILE_KEYS="input_audio,audio_mask"
        LORA_TARGET_MODULES="audio_attn1.to_k,audio_attn1.to_q,audio_attn1.to_v,audio_attn1.to_out.0,audio_attn2.to_k,audio_attn2.to_q,audio_attn2.to_v,audio_attn2.to_out.0,audio_ff.net.0.proj,audio_ff.net.2"
        ;;
    a2a_ic_lora)
        DATA_FILE_KEYS="input_audio,reference_audio"
        LORA_TARGET_MODULES="audio_attn1.to_k,audio_attn1.to_q,audio_attn1.to_v,audio_attn1.to_out.0,audio_attn2.to_k,audio_attn2.to_q,audio_attn2.to_v,audio_attn2.to_out.0,audio_ff.net.0.proj,audio_ff.net.2"
        ;;
    av2av_ic_lora)
        DATA_FILE_KEYS="video,input_audio,reference_video,reference_audio"
        LORA_TARGET_MODULES="to_k,to_q,to_v,to_out.0"
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
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.5-${MODE}-${VARIANT}-lora-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

accelerate launch examples/ltx2/model_training/train_ltx25.py \
  --dataset_base_path "./models/train/LTX2.5-${MODE}-${VARIANT}-lora-cache" \
  --data_file_keys "$DATA_FILE_KEYS" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --dataset_repeat "$DATASET_REPEAT" \
  --ltx25_model_root "$MODEL_ROOT" \
  --ltx25_variant "$VARIANT" \
  --training_mode_config "$MODE_CONFIG" \
  --learning_rate 1e-4 \
  --num_epochs "$NUM_EPOCHS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.5-${MODE}-${VARIANT}-lora" \
  --lora_base_model "dit" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --enable_tensorboard_log \
  --task "sft:train"
