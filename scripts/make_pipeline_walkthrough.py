"""
Generates assets/pipeline_walkthrough/ -- a stage-by-stage visual run of the
detection + recognition pipeline on the full synthetic sample image, so the
README can show actual intermediate output instead of just describing the
steps.

Stages captured:
  0_original.png      -- data/sample_images/synth_00001.png, unmodified
  1_lines_removed.png -- grayscale + inverted Otsu threshold, then
                          remove_lines() strips grid/wall lines
  2_dilated.png        -- after directional H+V dilation (the "smear")
  3_recognized.png    -- final filtered/tightened boxes with the CRNN's
                          predicted text drawn on top of the original image

Run: python scripts/make_pipeline_walkthrough.py
"""
import os
import sys
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from model.infer import CRNNWithRNN
from detection.classical_cv import remove_lines

SOURCE_IMAGE = "data/sample_images/synth_00001.png"
OUT_DIR = "assets/pipeline_walkthrough"
MODEL_PATH = "model/crnn_epoch14.pth"

characters = "½¾¼0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-'\" "
char_to_idx = {c: i + 1 for i, c in enumerate(characters)}
idx_to_char = {v: k for k, v in char_to_idx.items()}


def load_ocr_model(device):
    model = CRNNWithRNN(img_channel=3, num_classes=len(characters) + 1).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(img, model, device):
    target_height = 32
    aspect_ratio = img.width / img.height if img.height > 0 else 1.0
    target_width = max(int(target_height * aspect_ratio), 8)
    img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    MAX_ALLOWED_WIDTH = 800
    if img_tensor.shape[3] > MAX_ALLOWED_WIDTH:
        img_tensor = F.interpolate(img_tensor, size=(32, MAX_ALLOWED_WIDTH),
                                    mode="bilinear", align_corners=False)
    with torch.no_grad():
        preds = model(img_tensor)
        preds = preds.permute(1, 0, 2)
        preds_idx = preds.argmax(dim=2).permute(1, 0).cpu().numpy()[0]
        pred_text = []
        prev = -1
        for idx in preds_idx:
            if idx != prev and idx != 0:
                pred_text.append(idx_to_char.get(idx, ""))
            prev = idx
        return "".join(pred_text)


def predict_crop(crop_img, model, device):
    if crop_img.width >= crop_img.height * 1.1:
        return run_inference(crop_img, model, device)
    crop_rotated = crop_img.copy().rotate(-90, expand=True)
    return run_inference(crop_rotated, model, device)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 0. The full floor plan, unmodified
    full_bgr = cv.imread(SOURCE_IMAGE)
    cv.imwrite(f"{OUT_DIR}/0_original.png", full_bgr)

    full_gray = cv.cvtColor(full_bgr, cv.COLOR_BGR2GRAY)

    # Otsu threshold (inverted so ink = white, matches detection/classical_cv.py)
    _, thresh = cv.threshold(full_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # 1. Remove grid/wall lines (same function the real pipeline uses)
    no_lines = remove_lines(thresh)
    cv.imwrite(f"{OUT_DIR}/1_lines_removed.png", no_lines)

    # 2. Directional dilation (H + V) -- this is the "smearing" step that
    #    bridges individual letters into solid word-blobs
    h_kernel = np.ones((3, 35), np.uint8)
    h_dilation = cv.dilate(no_lines, h_kernel, iterations=1)
    v_kernel = np.ones((35, 3), np.uint8)
    v_dilation = cv.dilate(no_lines, v_kernel, iterations=1)
    combined = cv.bitwise_or(h_dilation, v_dilation)
    cv.imwrite(f"{OUT_DIR}/2_dilated.png", combined)

    # Contours + the same aspect-ratio/size/density filter as run_cv(),
    # then tighten the padding -- not saved as its own stage image, just
    # used to produce the final boxes for the recognition step below
    contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detected_boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = max(w, h) / float(min(w, h))
        if ((w > 15 and h > 50) or (h > 15 and w > 50)) and aspect_ratio > 1.1:
            roi = no_lines[y:y + h, x:x + w]
            density = cv.countNonZero(roi) / (w * h)
            if density > 0.05:
                pad = 17
                new_x, new_y = x + pad, y + pad
                new_w = max(1, w - 2 * pad)
                new_h = max(1, h - 2 * pad)
                detected_boxes.append((new_x, new_y, new_w, new_h))
    print(f"{len(detected_boxes)} boxes survived filtering")

    # 3. Run the CRNN recognizer on each box and draw boxes + predicted text
    #    on top of the original image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ocr_model(device)
    full_pil = Image.open(SOURCE_IMAGE).convert("RGB")

    recognized_img = full_bgr.copy()
    for (x, y, w, h) in detected_boxes:
        text_crop = full_pil.crop((x, y, x + w, y + h))
        text = predict_crop(text_crop, model, device)
        cv.rectangle(recognized_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(recognized_img, text, (x, max(y - 6, 12)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.imwrite(f"{OUT_DIR}/3_recognized.png", recognized_img)

    print(f"Wrote stage images to {OUT_DIR}/")


if __name__ == "__main__":
    main()
