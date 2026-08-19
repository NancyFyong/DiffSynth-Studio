# Full-resolution LTX-2.5 inference samples for PR #1602

This fork-only media set is referenced from [modelscope/DiffSynth-Studio PR #1602](https://github.com/modelscope/DiffSynth-Studio/pull/1602). It contains only full-resolution, complete LTX-2.5 inference results represented by implementation commit `9b75f73d1d3a7e095012568122a086102ab6bfa8`.

Every MP4 is 960×576, 121 frames at 24 FPS (5.041667 seconds), with H.264 video and 48 kHz stereo AAC audio. Each `-preview.gif` is a compact visual preview without audio. The P0 Distilled T2AV result uses `LTX25_VRAM_LIMIT_GB=16`.

## Included modes

- P0 Distilled T2AV
- P1 Distilled I2AV
- Dev one-stage T2AV
- Dev one-stage I2AV
- Dev two-stage T2AV
- Dev two-stage I2AV
- A2V
- Retake
- Keyframe interpolation
- Pixel Spatial Upscaler IC-LoRA

`manifest.json` records SHA-256, codec metadata, duration, and the implementation commit for every result. `SHA256SUMS` covers every MP4 and GIF. This set intentionally excludes all 64×64 smoke media.
