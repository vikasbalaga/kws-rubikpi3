# Key Word Spotting (KWS) live streaming demo on RUBIK Pi 3 development board

The project contains Python utilities for generating a robust speech dataset and running the trained KWS model.

The KWS model generation has been done using [Edge Impulse KWS tutorials]([url](https://docs.edgeimpulse.com/tutorials/end-to-end/keyword-spotting))

## Project Structure

```text
.
├── README.md
├── requirements.txt
│
└── python/
    ├── <dataset_generation_script>.py
    ├── <training_or_processing_script>.py
    ├── <kws_demo_script>.py
    │
    └── model_files/
        └── kws_model.h5
```

The `python/model_files/` directory contains the trained KWS model.

## Python Scripts

### 1. Dataset Generation

Generates a robust KWS dataset with three classes:

* **Wake Word** – wake-word samples generated using Edge TTS and optional real voice recordings.
* **Unknown** – general speech samples generated from the LibriSpeech-PC text corpus.
* **Noise** – real-world noise from MS-SNSD with synthetic noise as a fallback.

The generated speech can be augmented using:

* Different TTS voices
* Speech rate variations
* Pitch variations
* Praat-based pseudo-speaker transformations
* Reverb
* Gain variations
* Background noise

Example:

```bash
python python/generate_wakeword_dataset.py \
    --wake-word "Hey Ruby" \
    --out-dir dataset \
    --my-voice-dir my_voice \
    --wake-clips 900 \
    --unknown-clips 900 \
    --noise-clips 400 \
    --pseudo-speakers-per-recording 25
```

For a quick test:

```bash
python python/generate_wakeword_dataset.py \
    --wake-word "Hey Ruby" \
    --out-dir dataset \
    --wake-clips 10 \
    --unknown-clips 10 \
    --noise-clips 10
```

### 2. Model Export to ONNX

This script is responsible for exporting the Keras (.h5) or tflite models generated from Edge impulse into ONNX FP32 format.

Example:

```bash
python gen_onnx.py --input kws_model.keras --output kws_model.onnx
```

See the script's command-line help for available options:

```bash
python gen_onnx.py --help
```

### 3. KWS Inference Demo

Runs the trained KWS model for inference. The demo supports:

* Live microphone inference
* 16 kHz WAV file inference
* 48 kHz WAV file inference

The audio is processed using the Edge Impulse-compatible MFCC pipeline before being passed to the KWS model.

Example — live inference:

```bash
python python/kws_demo_rubikpi3.py \
    --model kws_model.onnx
```

Example — 16 kHz WAV:

```bash
python python/kws_demo_rubikpi3.py \
    --model kws_model.onnx \
    --wav test.wav
```

Example — 48 kHz WAV:

```bash
python python/kws_demo_rubikpi3.py \
    --model kws_model.onnx \
    --wav_48k test.wav
```

To change the wake-word detection threshold:

```bash
python python/kws_demo_rubikpi3.py \
    --model kws_model.onnx \
    --threshold 0.8
```

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The dataset generation workflow also requires **FFmpeg** to be installed and available in the system `PATH`.

## Dataset Classes

The generated dataset follows this structure:

```text
dataset/
├── wake_word/
├── unknown/
├── noise/
└── manifests/
```

All generated audio clips are normalized to:

* **Sample rate:** 16 kHz
* **Channels:** Mono
* **Duration:** 1 second

## Model

The trained model is stored under:

```text
python/model_files/kws_model.h5
```

The model is intended for three-class keyword spotting:

```text
0 - Wake Word
1 - Noise
2 - Unknown
```

## Notes

* The unknown class uses automatically sourced general speech rather than manually maintained sentences.
* The generated dataset is designed to improve robustness against speaker, pronunciation, background noise, and acoustic-environment variations.
