# Warehouse Floorplan OCR

A two-stage computer vision pipeline that reads text directly off scanned warehouse floor plans — shelf labels, bin IDs, dimension callouts, room labels — with no manual annotation needed at inference time. Classical CV finds where the text is; a custom-trained CRNN (CNN + bidirectional LSTM + CTC loss) reads what it says.

**Note:** this project was built during a summer internship, trained and tested on a real company floor plan. I don't hold a license to that image or own the rights to redistribute it, so it — and anything derived from it (extracted text, annotated renders) — is intentionally excluded from this repo. The demo below runs on a synthetically generated floor plan instead, using the same pipeline and the same trained model.

## Example output

Full pipeline run on a synthetic floor-plan-style image (3200×3200px), fully automated — detection and recognition, no manual boxes:

![Verification sample](outputs/verification_sample.jpg)

Green boxes are detected text regions (classical CV); red text is the model's prediction for each region. 59 regions detected and transcribed in one pass, including rotated/vertical text and a mix of alphanumeric codes and dimension strings.

## Pipeline

1. **Text detection (classical CV, `detection/classical_cv.py`)** — adaptive thresholding + morphological line removal (so warehouse grid/wall lines don't get mistaken for text) + directional dilation + contour detection, with an aspect-ratio and ink-density filter to reject noise. No learned model needed for this stage — it works directly on any floor plan's line-art style.
2. **Text recognition (CRNN, `model/infer.py`)** — each detected box is cropped, resized to a fixed height, and passed through a VGG-style CNN backbone feeding a 2-layer bidirectional LSTM, trained with CTC loss so it doesn't need per-character alignment at training time. Handles variable-length, variable-orientation text (vertical crops are rotated before inference).

## Results

**Classical CV detector** (evaluated against 400 hand-labeled ground-truth boxes from the original training data):
- Precision: 0.99
- Recall: 0.93

**CRNN recognizer** (evaluated on a held-out validation split of the synthetic training set):
- Validation CER (character error rate): 0.0033
- Validation Loss: 0.0132

Training ran 25 epochs on two Kaggle T4 GPUs (with mixed precision + DataParallel); a dataloader worker crash interrupted it at epoch 17, but per-epoch checkpointing meant no progress was lost — epoch 14 had the best validation CER and is the weights file shipped here (`model/crnn_epoch14.pth`).

## Repository layout

```
detection/        classical CV text-region detector (no training needed)
model/            CRNN architecture + trained weights + standalone inference (infer.py)
evaluation/       end-to-end pipeline scripts: run detection+recognition on a sample image,
                  then draw the predictions back onto the image for visual verification
data/             a synthetic sample floor-plan-style image for the demo above
outputs/          example pipeline output (detections CSV + the annotated verification image)
notebooks/        the original Kaggle notebook — full training run, CTC decoding, loss/CER
                  curves, and dataset generation code (proprietary-data cells cleared, see below)
```

## Training data

The CRNN was trained on a large synthetic dataset of generated text crops rendered in warehouse-floorplan-style fonts and layouts (not included here — see the notebook's dataset-generation cells), and validated against both a synthetic held-out split and real, hand-labeled floor plan images from the original internship project. Those real images and their derived outputs aren't mine to redistribute, so they're not included in this repo — see the licensing note at the top.

## Reproducing

```
pip install -r requirements.txt
python -m evaluation.test_main_image          # full pipeline: detect + recognize, writes the CSV
python evaluation/verify_results_main_image.py # draws the predictions back onto the image for visual check
```

Run from the repo root (the evaluation scripts import `model` and `detection` as packages). This regenerates `outputs/sample_ocr_detections.csv` and `outputs/verification_sample.jpg` from scratch using the included synthetic sample image and trained weights.

For the standalone recognizer only (e.g. to run on your own pre-cropped text images):
```python
from model.infer import predict_text
from PIL import Image

text = predict_text(Image.open("your_crop.png"))
```

The full training run, including the synthetic dataset generator and the Kaggle-specific GPU training loop, is in `notebooks/warehouse-ocr.ipynb`. A couple of cells that displayed the real company floor plan's layout have had their outputs cleared (source code is unchanged) — see the licensing note at the top.

## Known limitations

- Character confusions between visually similar glyphs (e.g. `O` vs `0`) occasionally show up in dense alphanumeric codes — a known, common OCR failure mode, not specific to this model.
- The classical CV detector is tuned for floor-plan-style line art (clean vector/scanned lines, moderate text density) — it has not been validated on noisier scan types (handwriting, low-contrast photocopies).
- Trained and validated primarily on one warehouse's floor plan style; generalization to very different floor-plan conventions (different label formats, font styles) is untested.
