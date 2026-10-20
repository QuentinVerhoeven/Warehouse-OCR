#This script generates much more blurry images, lines going through text, etc to train the model on harder images

import os
import random
import json
import string
import glob
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Output paths
OUTPUT_IMAGE_PATH = "../images_hard"
OUTPUT_LABELS_PATH = "../labels_hard"
os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
os.makedirs(OUTPUT_LABELS_PATH, exist_ok=True)

IMAGE_SIZE = 3200

PALETTES = [(215, 230, 250), (225, 240, 255), (230, 230, 230), (255, 255, 255), (255, 245, 235)]


def boxes_overlap(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        # Returns True if boxes intersect in both horizontal and vertical axes
        return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)


def get_fonts():
    #Very cool, dynamically locate system fonts for variety in training data
    font_paths = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)

    good_fonts = ['sans', 'mono', 'liberation', 'dejavu', 'arial', 'freesans', 'freemono', 'courier']
        
    # Exclude handwriting, symbols, serif, decorative, and slanted styles
    bad_fonts = [
        'symbol', 'dingbat', 'italic', 'oblique', 'serif', 'gothic', 'script', 
        'handwriting', 'comic', 'display', 'decorative', 'blackletter', 'bolditalic'
    ]
    clean_fonts = []


    for path in font_paths:
        path_lower = path.lower()
        # Must contain at least one CAD keyword, and no excluded keywords
        matches_cad = any(kw in path_lower for kw in good_fonts)
        has_excluded = any(kw in path_lower for kw in bad_fonts)
        
        if matches_cad and not has_excluded:
            clean_fonts.append(path)


    return clean_fonts

def load_random_font():
    fonts = get_fonts()
    font_size = int(IMAGE_SIZE * .01)

    if fonts:
        try:
            return ImageFont.truetype(random.choice(fonts), font_size)
        except Exception:
            print("FONT LOADING FAILED")
            pass
    return ImageFont.load_default()



def draw_background_lines(draw, width=IMAGE_SIZE, height=IMAGE_SIZE):
    #Draws random grey lines to mimic structural paths and walls.
    num_lines = random.randint(10, 25)
    for _ in range(num_lines):
        line_color = (180, 180, 180) # Light grey structural line
        

        x1,y1,x2,y2 = 0,0,0,0
        # Pick random start and end points on the canvas, warehouse lines are almost always in the cardinal directions though 
        x1, y1 = random.randint(0, width), random.randint(0, height)
        if random.choice([0,1]) == 0: #horizontal
            x2, y2 = random.randint(0, width), y1
        else: #vertical
            x2, y2 = x1, random.randint(0, height)
        
        
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=random.choice([2, 3, 4, 5, 6]))





def generate_label():
    length = random.randint(2, 10)
    chars = string.ascii_letters + string.digits
    text = ''.join(random.choice(chars) for _ in range(length))

    if random.random() > 0.8 and length >= 3:
        insert_position = random.randint(1, length - 1)
        text = text[:insert_position] + '-' + text[insert_position:]
    return text









def draw_single_rack_bay(img, draw, box, text_angle, text, box_color, outline_color, annotations):
    x1, y1, x2, y2 = box

    #Draw the bay boundary box
    draw.rectangle([x1, y1, x2, y2], fill=box_color, outline=outline_color)

    #Load font and measure the label that was passed in
    font = load_random_font()
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top

    box_w = x2 - x1
    box_h = y2 - y1

    if text_angle != 0:
        txt_img = Image.new('RGBA', (text_w, text_h), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((-left, -top), text, fill=(0, 0, 0), font=font)

        rotated_txt = txt_img.rotate(text_angle, expand=True)

        paste_x = x1 + (box_w - text_h) // 2
        paste_y = y1 + (box_h - text_w) // 2
        img.paste(rotated_txt, (paste_x, paste_y), rotated_txt)

        annotations.append({
            "label": text,
            "points": [[paste_x, paste_y], [paste_x + text_h, paste_y + text_w]],
            "shape_type": "rectangle",
            "angle": text_angle
        })
    else:
        text_x = x1 + (box_w - text_w) // 2 - left
        text_y = y1 + (box_h - text_h) // 2 - top
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        annotations.append({
            "label": text,
            "points": [[text_x + left, text_y + top], [text_x + left + text_w, text_y + top + text_h]],
            "shape_type": "rectangle",
            "angle": 0
        })


def draw_rack_row(img, draw, existing_boxes, annotations):
    is_vertical_row = random.choice([True, False])
    is_double_row = random.choice([True, False])
    font = load_random_font()

    num_bays = random.randint(4, 8)
    # Generate independent labels for each row
    labels1 = [generate_label() for _ in range(num_bays)]
    labels2 = [generate_label() for _ in range(num_bays)] if is_double_row else []

    padding_along = random.randint(60, 150)
    padding_across = random.randint(20, 120)

    # Size each bay using both sets of labels so the back-to-back bays align perfectly
    bay_dims = []
    for text in (labels1 + labels2):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        if is_vertical_row:
            bay_dims.append((text_h + 2 * padding_across, text_w + 2 * padding_along))
        else:
            bay_dims.append((text_w + 2 * padding_along, text_h + 2 * padding_across))

    max_w = max(d[0] for d in bay_dims)
    max_h = max(d[1] for d in bay_dims)

    spacing = random.choice([1, 4, 8])
    back_to_back_spacing = random.choice([1, 5, 10])

    if is_vertical_row:
        row_w = max_w if not is_double_row else (2 * max_w + back_to_back_spacing)
        row_h = num_bays * max_h + (num_bays - 1) * spacing
    else:
        row_w = num_bays * max_w + (num_bays - 1) * spacing
        row_h = max_h if not is_double_row else (2 * max_h + back_to_back_spacing)

    max_attempts = 200
    row_placed = False
    start_x = start_y = 0

    for _ in range(max_attempts):
        margin = 40

        start_x = random.randint(50, IMAGE_SIZE - 50)
        start_y = random.randint(50, IMAGE_SIZE - 50)
        row_box = (start_x - margin, start_y - margin, start_x + row_w + margin, start_y + row_h + margin)

        if row_box[2] > IMAGE_SIZE - 20 or row_box[3] > IMAGE_SIZE - 20:
            continue

        if not any(boxes_overlap(row_box, eb) for eb in existing_boxes):
            existing_boxes.append(row_box)
            row_placed = True
            break

    if not row_placed:
        return False

    outline_color = random.choice([(0, 0, 0), (100, 100, 100), None])

    box_color1 = random.choice(PALETTES)
    text_angle1 = random.choice([90, 270]) if is_vertical_row else 0

    box_color2 = random.choice(PALETTES)
    text_angle2 = random.choice([90, 270]) if is_vertical_row else 0

    def draw_single_line(sx, sy, line_labels, line_angle, line_color):
        for i in range(num_bays):
            if is_vertical_row:
                bay_x1 = sx
                bay_y1 = sy + i * (max_h + spacing)
                bay_x2 = bay_x1 + max_w
                bay_y2 = bay_y1 + max_h
            else:
                bay_x1 = sx + i * (max_w + spacing)
                bay_y1 = sy
                bay_x2 = bay_x1 + max_w
                bay_y2 = bay_y1 + max_h

            draw_single_rack_bay(
                img, draw, (bay_x1, bay_y1, bay_x2, bay_y2),
                line_angle, line_labels[i], line_color, outline_color, annotations
            )

    draw_single_line(start_x, start_y, labels1, text_angle1, box_color1)

    if is_double_row:
        if is_vertical_row:
            draw_single_line(start_x + max_w + back_to_back_spacing, start_y, labels2, text_angle2, box_color2)
        else:
            draw_single_line(start_x, start_y + max_h + back_to_back_spacing, labels2, text_angle2, box_color2)

    return True






def draw_dimension_line(img, draw, existing_boxes, annotations):
    font = load_random_font()
    text = generate_label()
    # force a dimension-shaped string for this call, not any random category
    ft, inch = random.randint(0, 150), random.randint(0, 11)
    num, denom = random.randint(1, 12), random.randint(1, 12)
    fmt = random.choice(["{}'-{}\"", "{}' {}\"", "{}'", "{}'-{} {}/{}\"", "{}'- {} {}/{}\"", "{}' -{} {}/{}\""])
    text = fmt.format(ft, inch, num, denom)

    is_vertical = random.choice([True, False])
    line_len = random.randint(300, 1200)
    tick_size = random.randint(15, 30)

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top

    max_attempts = 100
    for _ in range(max_attempts):
        if is_vertical:
            x1 = random.randint(50, IMAGE_SIZE - 50)
            y1 = random.randint(50, IMAGE_SIZE - line_len - 50)
            y2 = y1 + line_len
            bbox_region = (x1 - text_h - 20, y1 - 10, x1 + 10, y2 + 10)
        else:
            y1 = random.randint(50, IMAGE_SIZE - 50)
            x1 = random.randint(50, IMAGE_SIZE - line_len - 50)
            x2 = x1 + line_len
            bbox_region = (x1 - 10, y1 - text_h - 20, x2 + 10, y1 + 10)

        if not any(boxes_overlap(bbox_region, eb) for eb in existing_boxes):
            existing_boxes.append(bbox_region)
            break
    else:
        return False

    line_color = (100, 100, 100)

    if is_vertical:
        draw.line([(x1, y1), (x1, y2)], fill=line_color, width=3)
        draw.line([(x1 - tick_size, y1), (x1 + tick_size, y1)], fill=line_color, width=3)
        draw.line([(x1 - tick_size, y2), (x1 + tick_size, y2)], fill=line_color, width=3)

        txt_img = Image.new('RGBA', (text_w, text_h), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((-left, -top), text, fill=(0, 0, 0), font=font)
        rotated_txt = txt_img.rotate(90, expand=True)
        paste_x = x1 - text_h - 15
        paste_y = (y1 + y2) // 2 - text_w // 2
        img.paste(rotated_txt, (paste_x, paste_y), rotated_txt)

        annotations.append({
            "label": text,
            "points": [[paste_x, paste_y], [paste_x + text_h, paste_y + text_w]],
            "shape_type": "rectangle",
            "angle": 90
        })
    else:
        draw.line([(x1, y1), (x2, y1)], fill=line_color, width=3)
        draw.line([(x1, y1 - tick_size), (x1, y1 + tick_size)], fill=line_color, width=3)
        draw.line([(x2, y1 - tick_size), (x2, y1 + tick_size)], fill=line_color, width=3)

        text_x = (x1 + x2) // 2 - text_w // 2 - left
        text_y = y1 - text_h - 15
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        annotations.append({
            "label": text,
            "points": [[text_x + left, text_y + top], [text_x + left + text_w, text_y + top + text_h]],
            "shape_type": "rectangle",
            "angle": 0
        })

    return True












def draw_interference_lines(img, draw):
    #Draws long dimension lines 
    for _ in range(random.randint(3, 7)):
        line_color = (80, 80, 80)
        if random.choice([True, False]):
            y = random.randint(100, IMAGE_SIZE - 100)
            draw.line([(0, y), (IMAGE_SIZE, y)], fill=line_color, width=2)
            draw.line([(0, y-15), (0, y+15)], fill=line_color, width=3)
            draw.line([(IMAGE_SIZE, y-15), (IMAGE_SIZE, y+15)], fill=line_color, width=3)
        else:
            x = random.randint(100, IMAGE_SIZE - 100)
            draw.line([(x, 0), (x, IMAGE_SIZE)], fill=line_color, width=2)
            draw.line([(x-15, 0), (x+15, 0)], fill=line_color, width=3)
            draw.line([(x-15, IMAGE_SIZE), (x+15, IMAGE_SIZE)], fill=line_color, width=3)

def create_advanced_image(index):
    # Setup base image
    bg_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
    img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)

   
    background_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
    img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=background_color)
    draw = ImageDraw.Draw(img)

    font = load_random_font()

    draw_background_lines(draw, width=IMAGE_SIZE, height=IMAGE_SIZE)

    existing_boxes, annotations = [],[]
    
    # Generate 4 to 8 rows of warehouse racks
    num_rows = random.randint(4, 8)
    for _ in range(num_rows):
        draw_rack_row(img, draw, existing_boxes, annotations)


    num_dims = random.randint(3, 8)
    for _ in range(num_dims):
        draw_dimension_line(img, draw, existing_boxes, annotations)



    # Add Architectural Interference
    draw_interference_lines(img, draw)



    # Blur
    blur_radius = random.uniform(.3, 2.5)
    if blur_radius > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Contrast
    if random.random() > 0.3:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
        
    # Brightness
    if random.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))



    img_filename = f"adv_synth_{index:05d}.png"
    json_filename = f"adv_synth_{index:05d}.json"

    # Save
    img.save(os.path.join(OUTPUT_IMAGE_PATH, f"adv_synth_{index:05d}.png"))

    #save labels in JSON file
    label_data = {
        "imagePath": img_filename,
        "shapes": annotations
    }
    with open(os.path.join(OUTPUT_LABELS_PATH, json_filename), 'w') as f:
        json.dump(label_data, f, indent=2)

    return img



if __name__ == "__main__":

    
    for i in range(3000): 
        create_advanced_image(i)

