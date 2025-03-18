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

os.makedirs(output_folder, exist_ok=True)


session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_height, original_width = img.shape[:2]
    ratio = min(input_size / original_width, input_size / original_height)
    new_unpad = (int(round(original_width * ratio)), int(round(original_height * ratio)))
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    dw = input_size - new_unpad[0]
    dh = input_size - new_unpad[1]
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    padded[top:top + new_unpad[1], left:left + new_unpad[0]] = resized
    padded = padded.astype(np.float32) / 255.0
    blob = padded.transpose(2, 0, 1)[np.newaxis, ...]
    outputs = session.run(None, {input_name: blob})[0][0]
    txt_lines = []
    for det in outputs:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < conf_threshold:
            continue
        x1 = (x1 - left) / ratio
        y1 = (y1 - top) / ratio
        x2 = (x2 - left) / ratio
        y2 = (y2 - top) / ratio

        x1 = np.clip(x1, 0, original_width)
        y1 = np.clip(y1, 0, original_height)
        x2 = np.clip(x2, 0, original_width)
        y2 = np.clip(y2, 0, original_height)

        cx = (x1 + x2) / 2 / original_width
        cy = (y1 + y2) / 2 / original_height
        w = (x2 - x1) / original_width
        h = (y2 - y1) / original_height

        txt_lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    output_path = os.path.join(output_folder, Path(image_file).stem + ".txt")
    with open(output_path, 'w') as f:
        f.write('\n'.join(txt_lines))