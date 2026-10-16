import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import os
import csv

# Import your local model architecture
from model.infer import CRNNWithRNN
from detection.classical_cv import run_cv

# Configurations
MODEL_PATH = "model/crnn_epoch14.pth" #the best epoch from training
IMAGE_PATH = "data/sample_images/synth_00001.png"

# Corrected destination path
OUTPUT_DIR = "outputs"
OUTPUT_CSV = f"{OUTPUT_DIR}/sample_ocr_detections.csv"

# CTC Decoding setup
characters = "½¾¼0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-'\" "
char_to_idx = {char: i + 1 for i, char in enumerate(characters)}
idx_to_char = {v: k for k, v in char_to_idx.items()}


def load_ocr_model(device):
    print(f"Loading OCR weights from {MODEL_PATH}...")
    model = CRNNWithRNN(img_channel=3, num_classes=len(characters) + 1).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Strip DataParallel wrapper if present
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model

def run_inference(img, model, device):
    
    target_height = 32
    aspect_ratio = img.width / img.height if img.height > 0 else 1.0
    target_width = max(int(target_height * aspect_ratio), 8)
    img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    MAX_ALLOWED_WIDTH = 800
    if img_tensor.shape[3] > MAX_ALLOWED_WIDTH:
        img_tensor = F.interpolate(
            img_tensor, size=(32, MAX_ALLOWED_WIDTH), 
            mode='bilinear', align_corners=False
        )
        
    with torch.no_grad():
        preds = model(img_tensor)
        preds = preds.permute(1, 0, 2) 
        
        # Calculate probabilities
        probs = torch.softmax(preds, dim=2)
        max_probs, preds_idx = probs.max(dim=2)
        
        # Convert to 1D arrays for iteration
        preds_idx = preds_idx.permute(1, 0).cpu().numpy()[0]
        max_probs = max_probs.permute(1, 0).cpu().numpy()[0]
        
        pred_text = []
        char_confidences = []
        prev = -1
        
        # Decode CTC and capture the confidence of ONLY the valid characters
        for i, idx in enumerate(preds_idx):
            if idx != prev and idx != 0:
                pred_text.append(idx_to_char.get(idx, ''))
                char_confidences.append(max_probs[i]) # Save the prob of this specific char
            prev = idx
            
        # The true confidence is the average probability of the characters, ignoring blanks
        confidence = sum(char_confidences) / len(char_confidences) if char_confidences else 0.0
        
        return "".join(pred_text), confidence

def predict_crop(crop_img, model, device):
    # If the text is generally horizontal, run it normally
    if crop_img.width >= crop_img.height * 1.1:
        text, _ = run_inference(crop_img, model, device)
        return text     
    else:
        # Standard drafting convention: rotate -90 degrees so bottom-to-top text becomes left-to-right
        crop_rotated = crop_img.copy().rotate(-90, expand=True)
        text, _ = run_inference(crop_rotated, model, device)
        return text

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = load_ocr_model(device)
    
    print("Running classical CV bounding box detection...")
    boxes = run_cv(IMAGE_PATH)
    print(f"Found {len(boxes)} text regions.")
    
    print(f"Loading full floorplan for cropping: {IMAGE_PATH}")
    full_image = Image.open(IMAGE_PATH).convert('RGB')
    
    ocr_detections = []
    
    print("Running OCR inference on cropped regions...")
    for (x, y, w, h) in boxes:
        crop = full_image.crop((x, y, x + w, y + h))
        text = predict_crop(crop, model, device)
        
        ocr_detections.append({
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "text": text
        })
        
    print(f"Exporting detections to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "width", "height", "text"])
        writer.writeheader()
        writer.writerows(ocr_detections)
        
    print("Pipeline complete!")
