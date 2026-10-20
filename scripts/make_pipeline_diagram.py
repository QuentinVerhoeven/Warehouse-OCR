"""
Generates assets/pipeline_diagram.png: a placeholder block diagram of the
OCR pipeline. Classical-CV detection is broken into its real sub-steps
(threshold, morphological line removal, directional dilation, contour
detection, filtering) as separate boxes. The orientation check is a single
box (not a decision diamond + two outcome boxes). The CRNN's internal
stages (CNN backbone, BiLSTM, linear classifier + CTC decode) are kept
separate.

Meant to be replaced with a hand-made version -- see README.md.

Run: python scripts/make_pipeline_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_W, FIG_H = 12.0, 18.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("#0f1115")
ax.set_facecolor("#0f1115")

COL_CV = "#1b2230"
COL_CV_EDGE = "#4f7cff"
COL_BRANCH = "#241b2e"
COL_BRANCH_EDGE = "#b06fe0"
COL_CRNN = "#1a2b22"
COL_CRNN_EDGE = "#3ecf7a"
COL_OUT = "#241f1a"
COL_OUT_EDGE = "#d8a857"
TEXT = "#e8ecf4"
SUBTEXT = "#9aa5bd"


def box(x, y, w, h, title, lines, face, edge, title_color="#ffffff"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="top",
             fontsize=10, fontweight="bold", color=title_color, zorder=3)
    body = "\n".join(lines)
    ax.text(x + w / 2, y + h / 2 - 0.1, body, ha="center", va="center",
             fontsize=7.8, color=SUBTEXT, zorder=3, linespacing=1.45)
    return (x, y, w, h)


def side(b, which):
    x, y, w, h = b
    return {
        "top": (x + w / 2, y + h), "bottom": (x + w / 2, y),
        "left": (x, y + h / 2), "right": (x + w, y + h / 2),
    }[which]


def arrow(p1, p2, color="#5b6478"):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                         linewidth=1.5, color=color, zorder=1)
    ax.add_patch(a)


def section_label(y, text, color):
    ax.text(FIG_W / 2, y, text, ha="center", va="center",
             fontsize=9.5, fontweight="bold", color=color)


ax.text(FIG_W / 2, FIG_H - 0.3, "Warehouse Floorplan OCR — Pipeline",
        ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT)
ax.text(FIG_W / 2, FIG_H - 0.62,
        "PLACEHOLDER — replace with a hand-made diagram",
        ha="center", va="top", fontsize=9, color="#d8a857", style="italic")

W = 7.6
X = (FIG_W - W) / 2
H = 0.85
GAP = 0.3

y = FIG_H - 1.7
b_input = box(X, y, W, H, "Floor-plan image", ["PNG, any scan or render"],
              COL_CV, COL_CV_EDGE)

y -= (H + GAP + 0.35)
section_label(y + H + 0.15, "1. Text Region Detection  (detection/classical_cv.py) — NO learned model", "#8fb4ff")

y -= H
b_thresh = box(X, y, W, H, "1a. Grayscale + Otsu threshold",
               ["binarizes ink vs. background, inverted so ink = white"],
               COL_CV, COL_CV_EDGE)
arrow(side(b_input, "bottom"), side(b_thresh, "top"))

y -= (H + GAP)
b_lines = box(X, y, W, H, "1b. Remove grid/wall lines  (remove_lines())",
              ["morphological opening (erosion -> dilation) isolates straight",
               "lines H & V separately, then subtracts them from the image"],
              COL_CV, COL_CV_EDGE)
arrow(side(b_thresh, "bottom"), side(b_lines, "top"))

y -= (H + GAP)
b_dilate = box(X, y, W, H, "1c. Directional dilation (H + V) + combine",
               ["horizontal (3x35) and vertical (35x3) kernels bridge letters",
                "into word-blobs; results merged with bitwise OR"],
               COL_CV, COL_CV_EDGE)
arrow(side(b_lines, "bottom"), side(b_dilate, "top"))

y -= (H + GAP)
b_contours = box(X, y, W, H, "1d. Contour detection  (cv.findContours)",
                 ["outlines each blob -> bounding box (x, y, w, h) per contour"],
                 COL_CV, COL_CV_EDGE)
arrow(side(b_dilate, "bottom"), side(b_contours, "top"))

y -= (H + GAP)
b_filter = box(X, y, W, H, "1e. Filter + tighten boxes",
               ["aspect ratio, min size, and ink-density filters reject noise;",
                "shrink padding to tighten each box around the actual text"],
               COL_CV, COL_CV_EDGE)
arrow(side(b_contours, "bottom"), side(b_filter, "top"))

y -= (H + GAP + 0.3)
b_orient = box(X, y, W, H, "2. Rotate vertical/rotated crops",
               ["if width < height x 1.1, rotate the crop -90° before recognition",
                "(the fix vs. RapidOCR, which misread most vertical rack labels)"],
               COL_BRANCH, COL_BRANCH_EDGE)
arrow(side(b_filter, "bottom"), side(b_orient, "top"))

y -= (H + GAP)
b_prep = box(X, y, W, H, "3. Preprocessing",
             ["resize to height=32px (preserve aspect ratio) -> normalize -> tensor"],
             COL_CV, COL_CV_EDGE)
arrow(side(b_orient, "bottom"), side(b_prep, "top"))

y -= 0.55
ax.plot([X, X + W], [y + 0.25, y + 0.25], color="#3ecf7a", linewidth=1, alpha=0.5)
section_label(y + 0.42, "CRNN  (model/infer.py) — learned", "#8ff2b7")

y -= H
b_cnn = box(X, y, W, H, "4. CNN Backbone  (CRNNBackbone)",
            ["5 VGG-style conv blocks (64->128->256->256->512->512 ch.)",
             "+ BatchNorm + ReLU + MaxPool, asymmetric pooling preserves width",
             "Output: [Batch, 512, 1, W'] — one feature vector per column"],
            COL_CRNN, COL_CRNN_EDGE, title_color="#8ff2b7")
arrow(side(b_prep, "bottom"), side(b_cnn, "top"))

y -= (H + GAP)
b_lstm = box(X, y, W, H, "5. Bidirectional LSTM  (2 stacked layers, hidden=256)",
             ["reads the CNN's feature sequence left-to-right AND right-to-left,",
              "so each character's prediction uses context from both sides",
              "Output: [W', Batch, 512]"],
             COL_CRNN, COL_CRNN_EDGE, title_color="#8ff2b7")
arrow(side(b_cnn, "bottom"), side(b_lstm, "top"))

y -= (H + GAP)
b_lin = box(X, y, W, H, "6. Linear Classifier + CTC Decode",
            ["nn.Linear maps each LSTM output to a probability over 65 classes",
             "(64 characters + 1 CTC blank token); decode: argmax per column ->",
             "collapse duplicates -> drop blanks -> final text (CTC LOSS is",
             "training-only; this decode is what runs at inference)"],
            COL_CRNN, COL_CRNN_EDGE, title_color="#8ff2b7")
arrow(side(b_lstm, "bottom"), side(b_lin, "top"))

y -= 0.3
ax.plot([X, X + W], [y + 0.85, y + 0.85], color="#3ecf7a", linewidth=1, alpha=0.5)

y -= H
b_out = box(X, y, W, H, "Output",
            ["CSV: x, y, width, height, text  +  annotated image",
             "(green boxes = detections, red text = recognized text)"],
            COL_OUT, COL_OUT_EDGE, title_color="#f2d98f")
arrow(side(b_lin, "bottom"), side(b_out, "top"))

ax.set_ylim(y - 0.3, FIG_H)
plt.tight_layout()

import os
os.makedirs("assets", exist_ok=True)
out_path = "assets/pipeline_diagram.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print(f"wrote {out_path}")
