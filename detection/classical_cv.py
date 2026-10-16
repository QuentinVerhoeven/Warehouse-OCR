# This is the classical CV text box drawer from the kaggle notebook

import cv2 as cv
import numpy as np

def remove_lines(image):
    #scale how long the line masks are based on image resolution
    height, width = image.shape[:2]
    
    if width <= 3200: #synthetic
        line_target = 15
    else:
        scale_factor = width / 7060 #resolution of store image
        line_target = max(5, int(25 * scale_factor))


    #creates a mask of only horizontal lines
    h_line_kernel = np.ones((1, line_target), np.uint8)
    h_lines = cv.morphologyEx(image, cv.MORPH_OPEN, h_line_kernel, iterations=2)

    #thicken lines to catch all of them
    h_lines = cv.dilate(h_lines, np.ones((3, 3), np.uint8), iterations=1)


    v_line_kernel = np.ones((line_target,1), np.uint8)
    v_lines = cv.morphologyEx(image, cv.MORPH_OPEN, v_line_kernel, iterations=2)
    v_lines = cv.dilate(v_lines, np.ones((3, 3), np.uint8), iterations=1)

    combined = cv.bitwise_or(h_lines, v_lines)

    #to get text only subtract the lines from the original
    text_only = cv.subtract(image, combined)

    #plt.figure(figsize=(40,24))
    #plt.imshow(text_only, cmap='gray')
    #plt.show()

    return text_only


def run_cv(file_path):
    img = cv.imread(file_path)
    if img is None:
        print(f"FAILED TO LOAD IMAGE AT: {file_path}")
        return []
    
    og_img = img.copy()
    img_gray = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    
    blocksize = 7 #size of block of neighboring pixels for threshold
    offsetC = 3 #positive offset helps filter out small noise
    
    
    
    #use mean adaptive threshold to convert image to binary (Gaussian is overkill and is more computatoinaly expensive)
    _, img_gray = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)    #We invert so dilation works
    
    #Detect lines and remove them
    
    img_gray = remove_lines(img_gray)
    
    
    
    #HORIZONTAL PASS
    h_kernel = np.ones((3,35),np.uint8)
    h_dilation = cv.dilate(img_gray, h_kernel, iterations=1)
    
    #VERTICAL PASS
    v_kernel = np.ones((35,3),np.uint8)
    v_dilation = cv.dilate(img_gray, v_kernel, iterations=1)
    
    #combine using bitwise or
    combined = cv.bitwise_or(h_dilation, v_dilation)
    
    
    #get outlines (contours)
    contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv. CHAIN_APPROX_SIMPLE)
    
    
    detected_boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
    
    
        #use a an aspect ratio filter to get rid of square noise (words are rectangles)
        aspect_ratio = max(w, h) / float(min(w, h))
    
        
        if ((w > 15 and h > 50) or (h > 15 and w > 50)) and aspect_ratio > 1.1:
    
                #add density check (noise is almost always less dense)
                roi = img_gray[y:y+h, x:x+w]
                density = cv.countNonZero(roi) / (w * h)
                if density > .05:
                    #new logic to shrink boxes tigher around text
                    pad = 17 #17 because our kernel is 35 pixels wide, so it will add 17 pixels when dilating
                
                    new_x = x + pad
                    new_y = y + pad
                    
                    # Prevent negative dimensions if the original box was somehow smaller than the padding
                    new_w = max(1, w - (2 * pad))
                    new_h = max(1, h - (2 * pad))
                    
                    cv.rectangle(og_img, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
                    detected_boxes.append([new_x, new_y, new_w, new_h])
            
    #plt.figure(figsize=(25, 15))
    #plt.imshow(combined, cmap='gray')
    #plt.show()
    
    #plt.figure(figsize=(30,30))
    #plt.imshow(cv.cvtColor(og_img, cv.COLOR_BGR2RGB))
    #plt.show()
    
    
    return detected_boxes
