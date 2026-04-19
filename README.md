# Eye-Blink to Morse Code Decoder

A real-time webcam application that converts eye blinks into Morse code and decoded text.

This repository is developed as a Computing Project subject project by Group 7.

## What This Repository Currently Contains

```text
morse-code-to-eye-blink-decoder-apps/
|- implementation.py
|- requirements.txt
|- README.md
`- runs/
   `- classify/
      `- nano_100/
         `- weights/
            `- best.pt
```

Notes:
- `face_landmarker.task` is not committed in this repository. The app downloads it automatically on first run.
- NLP correction model files are not committed locally; they are loaded from Hugging Face when enabled.

## Features (Implemented)

- Real-time face and eye landmark detection using MediaPipe Face Landmarker.
- Eye state classification using YOLO classification weights at `runs/classify/nano_100/weights/best.pt`.
- Hybrid confidence scoring (YOLO + EAR).
- Mandatory calibration flow before detection:
  - EAR open sampling
  - EAR closed sampling
  - Dot and dash blink calibration
- Morse decoding with configurable letter, word, and sentence gaps.
- Optional NLP smoothing with IndoBERT Seq2Seq (`ZerXXX/indobert-corrector`).
- Streamlit dashboard with live video, status, Morse sequence, decoded text, and NLP output.

## Requirements

- Python 3.8+
- Webcam
- Internet connection (first run) for:
  - MediaPipe face landmarker model download
  - Hugging Face model download (if NLP correction is enabled)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run implementation.py
```

## How to Use

1. Open the app with the command above.
2. In the sidebar, click `Begin Cal.`.
3. Complete calibration stages using `Next Step` when each stage has enough samples:
   - EAR open
   - EAR closed
   - Dot and dash blinks
4. After calibration is complete, click `Start`.
5. Blink to generate Morse symbols and decoded text.
6. Optional controls:
   - `Clear Text`
   - `Remove ? (unresolved)`
   - `Stop`
   - `Reset`

## Main Settings in Sidebar

- Model:
  - `Use GPU`
- Confidence:
  - `Alpha (YOLO weight)`
  - `Blink Threshold`
- Timing:
  - `Letter Gap (seconds)`
  - `Word Gap (seconds)`
  - `Sentence Gap (seconds)`
- EAR normalization:
  - `EAR Min`
  - `EAR Max`
- NLP:
  - `Enable NLP Correction`
- Calibration:
  - `Blinks per type`
  - `EAR frames per state`

## Tech Stack

- Streamlit
- OpenCV
- MediaPipe
- Ultralytics YOLO
- PyTorch
- Hugging Face Transformers

## Important Repository Accuracy Notes

This README is intentionally aligned to the files currently present in this repository. It does not document training notebooks, extra YOLO model variants, or preprocessing scripts that are not included here.
