import copy
from typing import Iterable, List, Optional, Union
import dataclasses
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import open3d as o3d

# Default color palette for consistent visualization without supervision
DEFAULT_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 111), (26, 212, 255), (33, 138, 221), (31, 102, 255),
    (118, 106, 255), (161, 48, 185), (246, 64, 236), (255, 156, 249)
]

def vis_result_fast(
    image: np.ndarray, 
    detections, 
    classes, 
    color=None, 
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> np.ndarray:
    '''
    Annotate the image with detection results using native OpenCV.
    Supports sv.Detections objects or list-like detections.
    '''
    annotated_image = image.copy()
    
    # Extract detection data
    if hasattr(detections, 'xyxy'):
        xyxy = detections.xyxy
        mask = getattr(detections, 'mask', None)
        confidence = getattr(detections, 'confidence', None)
        class_id = getattr(detections, 'class_id', None)
    else:
        # Assume iterable of [x1, y1, x2, y2, confidence, class_id, ...]
        xyxy = np.array([d[0:4] for d in detections])
        confidence = np.array([d[4] for d in detections])
        class_id = np.array([d[5] for d in detections])
        mask = None

    palette = color if color is not None else DEFAULT_PALETTE

    for i in range(len(xyxy)):
        idx = i if instance_random_color else (class_id[i] if class_id is not None else 0)
        # groundingdino can leave class_id[i] as None when a phrase doesn't map back
        # to any class; fall back to instance index in that case.
        if idx is None:
            idx = i
        c = palette[int(idx) % len(palette)]
        # Convert (R, G, B) to (B, G, R) for OpenCV
        c_bgr = (c[2], c[1], c[0])

        x1, y1, x2, y2 = xyxy[i].astype(int)

        # 1. Draw Mask
        if mask is not None and mask[i] is not None:
            m = mask[i].astype(bool)
            roi = annotated_image[m]
            annotated_image[m] = (roi * 0.5 + np.array(c_bgr) * 0.5).astype(np.uint8)
            # Draw contours
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_image, contours, -1, c_bgr, 1)

        # 2. Draw BBox
        if draw_bbox:
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), c_bgr, 1)

            # 3. Draw Label
            if class_id is not None and classes is not None and class_id[i] is not None:
                cls_name = classes[int(class_id[i])]
                conf = confidence[i] if confidence is not None else 1.0
                label = f"{cls_name} {conf:.2f}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Label background
                cv2.rectangle(annotated_image, (x1, y1 - th - baseline), (x1 + tw, y1), c_bgr, -1)
                # Label text
                cv2.putText(annotated_image, label, (x1, y1 - baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated_image

def init_vis_image(goal_name, action = 0):
    vis_image = np.ones((537 * 3 + 100, 1165 * 2, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Observations" 
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Find {}  Action {}".format(goal_name, str(action))
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color
    
    vis_image = draw_box(vis_image, (1200, 50), 480, 480, color)
    return vis_image

def remove_image_border(vis_image: np.ndarray, black_bg: bool = False) -> np.ndarray:
    image_h, image_w = vis_image.shape[:2]
    gray = cv2.cvtColor(vis_image, cv2.COLOR_BGR2GRAY)
    if black_bg:
        _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(255-binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x += w//2
        y += h//2
        ratio = image_w / image_h
        if w / h > ratio:
            h = int(w / ratio)
        else:
            w = int(h * ratio)
        x = int(x - w / 2)
        y = int(y - h / 2)
        
        x = max(0, x)
        y = max(0, y)
        w = min(image_w - x, w)
        h = min(image_h - y, h)
        vis_image = vis_image[y:y+h, x:x+w]
        vis_image = cv2.resize(vis_image, (image_w, image_h), interpolation=cv2.INTER_AREA)
    return vis_image

def add_text(
    image: np.ndarray,
    text: str,
    position,
    font_scale: float = 1.0,
    color: tuple = (20, 20, 20),
    thickness: int = 2,
    vertical_align: str = 'top',
    horizontal_align: str = 'left',
) -> Union[np.ndarray, tuple]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize_w, textsize_h = cv2.getTextSize(text, font, font_scale, thickness)[0]

    if horizontal_align == 'center':
        position = (position[0] - textsize_w // 2, position[1])
    elif horizontal_align == 'right':
        position = (position[0] - textsize_w, position[1])
    if vertical_align == 'center':
        position = (position[0], position[1] + textsize_h // 2)
    elif vertical_align == 'top':
        position = (position[0], position[1] + textsize_h)

    if image is not None:
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        return image, (textsize_w, textsize_h)
    else:
        return (textsize_w, textsize_h)

def draw_box(image, start, h, w, color, thickness=1):
    x1, y1 = start
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(image, (x1-1, y1-1), (x2+1, y2+1), color, thickness)
    return image

def resize_with_height(image, height):
    scale = height / image.shape[0]
    new_width = int(image.shape[1] * scale)
    return [new_width, height]

def add_img(vis_image, img, start):
    x1, y1 = start
    h, w = img.shape[:2]
    x2, y2 = x1 + w, y1 + h
    x2 = min(x2, vis_image.shape[1])
    y2 = min(y2, vis_image.shape[0])
    vis_image[y1:y2, x1:x2] = img[:y2-y1, :x2-x1]
    return vis_image

def draw_text_with_circle(img, center, text):
    TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.35
    TEXT_THICKNESS = 1
    cv2.circle(img, center, 10, (0,0,0), TEXT_THICKNESS)
    text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (int(center[0] - text_size[0] / 2), int(center[1] - text_size[1] / 2))
    cv2.putText(img, text, text_origin, TEXT_FACE, TEXT_SCALE, (0,0,0), TEXT_THICKNESS, cv2.LINE_AA, bottomLeftOrigin=True)
    return img

def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat

def get_pyplot_as_numpy_img(img, cmap='plasma', figsize=(4.8, 4.8), origin='lower'):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img, cmap=cmap, origin=origin)
    if origin == 'upper':
        ax.yaxis.set_inverted(True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    ncols, nrows = fig.canvas.get_width_height()
    img_buffer = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)[..., :3]
    final_img = cv2.cvtColor(img_buffer, cv2.COLOR_RGB2BGR)
    plt.close()
    return final_img

def pad_frames_to_same_size(vis_frames):
    new_vis_Frames = []
    max_size = (0, 0)
    for frame in vis_frames:
        max_size = (max(max_size[0], frame.shape[0]), max(max_size[1], frame.shape[1]))
    for i, frame in enumerate(vis_frames):
        canvas = np.ones((max_size[0], max_size[1], 3), dtype=np.uint8) * 255
        h_gap = (max_size[0] - frame.shape[0])//2
        w_gap = (max_size[1] - frame.shape[1])//2
        canvas[h_gap:h_gap+frame.shape[0], w_gap:w_gap+frame.shape[1]] = frame
        new_vis_Frames.append(canvas) 
    return new_vis_Frames

def resize_and_pad(img, max_width, max_height, pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((max_height, max_width, 3), pad_color, dtype=np.uint8)
    x_offset = (max_width - new_w) // 2
    y_offset = (max_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

padding_w = 15
padding_h = 25
text_args = {"font_scale": 1.0, "color": (20, 20, 20), "thickness": 2}
caption_args = {"font_scale": 0.7, "color": (50, 50, 50), "thickness": 2, "horizontal_align": "center"}

def render_grid(vis_grid, img_size=(1920, 1080)):
    image_w = 0
    image_h = 0
    pos_x = padding_w
    pos_y = padding_h
    element_list = []
    for row in vis_grid:
        if row["type"] == "text":
            pos_x = padding_w
            row["pos"] = [pos_x, pos_y]
            textsize_w, textsize_h = add_text(None, row["text"], (0, 0), **row["args"])
            image_w = max(image_w, textsize_w)
            image_h += textsize_h + padding_h
            pos_y += textsize_h + padding_h
            element_list.append(row)
        elif row["type"] == "image_row":
            element_list.append(row)
            pos_x = padding_w
            total_width = padding_w * (len(row["image"]) - 1)
            caption_size_w, caption_size_h = add_text(None, "Caption", (0, 0), **caption_args)
            for i, (caption, img) in enumerate(row["image"]):
                new_size_w, new_size_h = resize_with_height(img, row["height"])
                total_width += new_size_w
                row["image"][i].append([pos_x, pos_y])
                row["image"][i].append([new_size_w, new_size_h])
                cap = {"type": "text", "args": caption_args, "text": caption, "pos": []}
                cap["pos"].append(pos_x + int(0.5*new_size_w))
                cap["pos"].append(pos_y + new_size_h + int(0.5*padding_h))
                element_list.append(cap)
                pos_x += new_size_w + padding_w
            image_w = max(image_w, total_width)
            pos_y += row["height"] + int(1.5*padding_h) + caption_size_h
            image_h += row["height"] + int(1.5*padding_h) + caption_size_h
    image_w += padding_w * 2
    image_h += padding_h * 2
    vis_image = np.ones((image_h, image_w, 3), dtype=np.uint8) * 255
    for row in element_list:
        if row["type"] == "text":
            add_text(vis_image, row["text"], row["pos"], **row["args"])
        elif row["type"] == "image_row":
            for i, (caption, img, pos, new_size) in enumerate(row["image"]):
                new_image = cv2.resize(img, new_size)
                vis_image = add_img(vis_image, new_image, pos)
    if img_size is not None:
        vis_image = resize_and_pad(vis_image, img_size[0], img_size[1])
    return vis_image
