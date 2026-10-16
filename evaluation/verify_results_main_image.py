import cv2
import csv
import os

# Configurations
IMAGE_PATH = "data/sample_images/synth_00001.png"
CSV_PATH = "outputs/sample_ocr_detections.csv"
OUTPUT_IMAGE = "outputs/verification_sample.jpg"

def verify():
    print(f"Loading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    
    if img is None:
        print("Failed to load image. Check your IMAGE_PATH.")
        return

    if not os.path.exists(CSV_PATH):
        print(f"Could not find CSV at {CSV_PATH}")
        return

    print(f"Drawing bounding boxes and text from {CSV_PATH}...")
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        
        for row in reader:
            x, y = int(row['x']), int(row['y'])
            w, h = int(row['width']), int(row['height'])
            text = row['text']
            
            # Draw a green bounding box around the text region
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw the predicted text in red just above the bounding box
            cv2.putText(
                img, text, (x, max(y - 5, 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            count += 1
            
    cv2.imwrite(OUTPUT_IMAGE, img)
    print(f"Done! Annotated {count} regions.")
    print(f"Open {OUTPUT_IMAGE} to visually verify the results.")

if __name__ == "__main__":
    verify()
