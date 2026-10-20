# pIAno — Visual-to-Music Generation with Mistral 7B + CLIP

## Overview

In this part of the project, we explore a multimodal AI pipeline that generates traditional Irish music in ABC notation from images. It combines a CLIP vision encoder (OpenCLIP ViT-B/32) with a fine-tuned Mistral 7B language model to translate visual content into playable musical scores, which are then synthesized into audio via FluidSynth.

The project runs on Kaggle with GPU acceleration (T4/P100).

---

## Architecture

```
Image (JPG/PNG)
      |
  OpenCLIP ViT-B/32          <-- Vision encoder (frozen)
      |
  512-dim embedding
      |
  Injected into Mistral       <-- Replaces <IMAGE_EMBED> token
  7B token stream
      |
  Mistral 7B + LoRA           <-- Fine-tuned on Irish music dataset
      |
  ABC Notation output
      |
  music21 + FluidSynth        <-- MIDI export and WAV synthesis
      |
  Playable audio
```

---

## Notebook Structure

### Cell 1 — Environment Setup
Installs all dependencies and imports. Configures Kaggle paths, clears GPU memory, disables TF32 for numerical stability, and loads the CLIP model onto the GPU.

**Key paths:**
- `SAVE_PATH`: `/kaggle/working/Mistral_Music_Project`
- `MODEL_ID`: `mistralai/Mistral-7B-v0.1`
- `DATA_FILE`: JSONL dataset file
- `MODEL_SAVE_PATH`: LoRA adapter output directory

### Cell 2 — Model Loading (Inference)
Loads the pIAno fine-tuned model from a Kaggle dataset input. Uses 4-bit NF4 quantization to fit Mistral 7B into GPU memory. Applies the LoRA adapters on top of the quantized base model and sets the model to evaluation mode.

### Cell 3 — Dataset Preparation (`prepare_vision_dataset`)
Builds the training dataset from the [sander-wood/irishman](https://huggingface.co/datasets/sander-wood/irishman) dataset. For each ABC tune:
- Extracts key, meter, and mode information
- Classifies tunes as "sad" or "happy" based on musical mode
- Generates a descriptive text prompt
- Encodes the prompt with CLIP text encoder to produce a 512-dim embedding
- Balances the dataset to equal numbers of sad and happy tunes
- Saves to a JSONL file with `text` and `image_features` fields

### Cell 4 — Model Setup for Training
Configures the base Mistral 7B model with 4-bit quantization, enables gradient checkpointing, and applies a LoRA configuration targeting all attention and MLP projection layers:

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |

### Cell 5 — Multimodal Collator and Training
Implements `MultimodalCollator`, which:
1. Tokenizes text inputs on CPU
2. Retrieves CLIP image embeddings from the dataset
3. Builds `inputs_embeds` by replacing the `<IMAGE_EMBED>` special token with the CLIP vector
4. Returns CPU tensors for the Trainer to handle GPU transfer

Training is run with `SFTTrainer` for 150 steps with a cosine LR schedule, `paged_adamw_8bit` optimizer, and gradient accumulation over 4 steps.

### Cell 6 — Learning Curve
Plots training loss over steps using matplotlib.

### Cell 7 — Music Generation Functions
Contains three core functions:

**`generer_musique_flexible(image_paths, weights)`**
Accepts one or more images with optional influence weights. Fuses CLIP embeddings as a weighted average, injects into Mistral, and generates ABC notation. Cleans illegal characters from the output.

**`nettoyer_abc(abc_string)`**
Parses and validates the generated ABC notation. Preserves existing headers (X, T, M, L, K, R, Z, C), fills in missing defaults, and reassembles the score in the correct header order.

**`exporter_midi(abc_string, output_name)`**
Converts the ABC score to MIDI using `music21`. Includes a two-level rescue mechanism: first attempts full parse, then falls back to notes-only extraction if the structure is invalid.

### Cell 8 — Interactive UI
Builds a Jupyter widget interface:
- File uploader (JPG/PNG/JPEG, multi-image)
- Dynamic influence sliders (shown when multiple images are uploaded)
- Generation button that runs the full pipeline: CLIP analysis → Mistral composition → MIDI export → WAV synthesis → audio playback

### Cell 9 — LoRA vs Base Mistral Comparison (`comparer_avec_et_sans_lora`)
Runs the same generation twice on a test image — once with LoRA adapters enabled, once with adapters disabled — and prints both ABC outputs side by side for qualitative comparison.

---

## Requirements

- Kaggle GPU environment (T4 or P100 recommended)
- Python 3.10+
- `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`
- `open_clip_torch`, `music21`, `Pillow`, `ipywidgets`
- `fluidsynth` system package + `pyfluidsynth`
- SoundFont: `/usr/share/sounds/sf2/FluidR3_GM.sf2`

---

## Dataset

The model is trained on the **irishman** dataset by sander-wood (HuggingFace), which contains traditional Irish tunes in ABC notation with control codes describing structure, meter, and mode. 10,000 examples are used, filtered and balanced to 50% major / 50% minor modes.

---

## LoRA Adapter

The fine-tuned adapter weights are stored at:
```
/kaggle/input/datasets/elosetrabut/piano-data
```
This directory must contain the adapter config, weights, and tokenizer files. The `local_files_only=True` flag prevents any online fallback.

---

## Output Format

Generated music is output in [ABC notation](https://abcnotation.com/), a text-based music format. Example:
```
X:1
T:AI Composition
M:4/4
L:1/8
K:Emin
|: e3 d B2 AG | FGAB cdef | ...
```

The ABC score is then converted to MIDI and rendered to WAV at 44100 Hz using the General MIDI soundfont.

---

## Known Limitations

- The CLIP embedding dimension (512) is injected into the first 512 dimensions of Mistral's 4096-dim embedding space. The remaining dimensions are zero-padded, which is a simple but effective bridge.
- The model generates valid ABC notation most of the time, but may occasionally produce structurally invalid scores. The rescue mechanism in `exporter_midi` handles the most common failure cases.
- Generation quality is sensitive to temperature. Values between 0.7 and 0.85 tend to produce the best results.
