# Retrace

Turn any video into a simple 3D version of the scene, then replay the camera's exact path through it, synced with the original video.

**Everything runs in your browser.** No uploads, no accounts, no API keys, no cost. The only downloads are open-source libraries and free models from Hugging Face, which are cached after the first run.

Live: https://swowanders.github.io/Retrace/

## How to use

1. Open the page in Chrome. Android works.
2. Pick a video and press **Build 3D street**. Keep the screen on while it processes.
3. Press ▶. Switch views with **Ride** (the camera's eye), **Chase** (behind the camera) or **Orbit** (look around freely).
4. Pick a style from the row of chips, and use ⚙ for palette, piece size, filters and export.

## Styles

| Style | What it is |
|---|---|
| ▣ Blocks | Cubes: small near the camera, bigger far away, so distant scenery stays solid |
| ◆ Tiles | Flat squares turned to follow each surface |
| ● Spheres | Low-poly gems |
| · Points | The raw 3D samples |
| ⌗ Wire | Joined points: a wireframe of the depth |
| ▤ Sheets | Continuous surfaces from key frames |
| ◎ 3D Video | Every frame as a moving 3D surface |

**Palettes:** Original (with optional posterize), C64, NES, PICO-8, Game Boy, CGA, Noir, Sepia, Autumn, Synthwave, Vaporwave, Blueprint, plus Thermal and Ice (coloured by distance) and Rainbow (coloured by height).

## Works with any video

| Video | Result |
|---|---|
| Walking, cycling, driving, drone, rooms | A 3D space with the camera path replayed |
| Edited video with cuts | Each shot is built as its own piece, side by side, with shot buttons |
| Still / tripod camera | **Diorama** ◉: a frozen 3D set with the moving parts playing inside it |
| Untrackable (dancing, animation, screen recordings, chaos) | **3D video stage** ◎: every frame plays as a moving 3D surface |
| People and vehicles | Cut out of the model and shown as a live 3D layer instead |

## Exports

- **Video**:
  - MP4: rendered frame by frame, so it's smooth even on a slow phone. 720p or 1080p, 30 fps, with the original audio (AAC, or Opus where AAC isn't available).
  - WebM: live screen capture.
- **3D model** (.glb): one object per shot, in the current style. Opens in Blender or any glTF viewer.
- **Camera path** (.json): per-frame position and rotation, plus shot info.

## Models and licences

- Depth, Standard: [Depth Anything V2 Small](https://huggingface.co/onnx-community/depth-anything-v2-small) (Apache-2.0).
- Depth, High: [Depth Anything V2 Base](https://huggingface.co/onnx-community/depth-anything-v2-base). Its weights are **CC-BY-NC-4.0 (non-commercial)**, so use Standard for anything monetized.
- People and vehicles: [SegFormer-B0 (ADE20K)](https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512).
- Runtime: [Transformers.js](https://huggingface.co/docs/transformers.js), [three.js](https://threejs.org) and [mp4-muxer](https://github.com/Vanilagy/mp4-muxer).

## How it works

1. **Frames**: sampled at the chosen fps. Depth comes from the model above (WebGPU, or WASM fallback).
2. **Camera motion**:
   1. Corner features are tracked with pyramidal Lucas–Kanade optical flow, seeded by a coarse global shift.
   2. A forward–backward check drops unreliable tracks.
   3. RANSAC PnP against the previous frame's depth gives the camera pose.
   4. The depth scale is aligned frame to frame.
3. **Shots**:
   - A colour-histogram jump plus lost tracking marks a cut.
   - Mostly lost tracking marks a stage shot.
   - A camera that barely moves marks a still shot, whose background is the per-pixel median over time.
4. **Scene**:
   - Points are levelled to the ground plane (RANSAC + PCA).
   - They're grouped into cells sized by each sample's footprint at its distance, in power-of-two steps.
   - Each style is rendered from those cells, the points, or the depth grids.

Scale is approximate, and the result is a stylised model rather than a survey.

## Development

`index.html` is generated, so edit the files in `dev/`:

- `dev/core.js`: tracking, pose, shots, cells, palettes and surfaces (pure JS, no dependencies)
- `dev/app.js`: UI, models, viewer and exports
- `dev/app.html`: page shell

```
python3 dev/build.py        # rebuild index.html
node tests/test-core.mjs    # synthetic-scene tests for the core
```
