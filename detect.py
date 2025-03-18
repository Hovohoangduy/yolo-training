import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path

input_folder = "images"
output_folder = "tmp"
model_path = "pricetag_15_03.onnx"
conf_threshold = 0.2
input_size = 640
class_colors = [(0, 255, 0), (255, 0, 0)]  # Màu RGB cho từng class

os.makedirs(output_folder, exist_ok=True)

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB để xử lý
    original_height, original_width = img_rgb.shape[:2]
    ratio = min(input_size / original_width, input_size / original_height)
    new_unpad = (int(original_width * ratio), int(original_height * ratio))
    resized = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    dw = input_size - new_unpad[0]
    dh = input_size - new_unpad[1]
    top, bottom = dh//2, dh - dh//2
    left, right = dw//2, dw - dw//2
    padded[top:top+new_unpad[1], left:left+new_unpad[0]] = resized
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    outputs = session.run(None, {input_name: blob})[0][0]

    for det in outputs:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < conf_threshold:
            continue

        x1 = int((x1 - left)/ratio)
        y1 = int((y1 - top)/ratio)
        x2 = int((x2 - left)/ratio)
        y2 = int((y2 - top)/ratio)
        
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))

        cls_id = int(cls_id)
        color = class_colors[cls_id]
        label = f"Class {cls_id} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, img)