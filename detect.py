import cv2
import numpy as np
import onnxruntime as ort
import os
import glob

# Đường dẫn đến mô hình và thư mục
model_path = "pricetag_12_03.onnx"
image_folder = "test/images"
output_folder = "tmp"
os.makedirs(output_folder, exist_ok=True)

# Load mô hình ONNX và kiểm tra đầu vào/ra
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_width, input_height = input_shape[2], input_shape[3]

# Kiểm tra đầu ra mô hình
outputs = session.get_outputs()
print("Model outputs info:", [{"name": out.name, "shape": out.shape} for out in outputs])

# Hàm tạo màu dựa trên class_id
def get_color(class_id):
    """Tạo màu sắc duy nhất cho từng class dựa trên HSV."""
    hue = (class_id * 50) % 180  # Hue trong OpenCV nằm trong [0, 179]
    saturation = 255
    value = 255
    hsv_color = np.uint8([[[hue, saturation, value]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    color = tuple(map(int, bgr_color[0][0]))
    return color

# Hàm preprocess
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_width, input_height))
    image = image / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# Hàm postprocess
def postprocess(outputs, image_shape):
    if len(outputs[0].shape) == 3:
        output = outputs[0][0]
    else:
        output = outputs[0].transpose(0, 2, 1).squeeze(0)

    boxes = []
    confidences = []
    class_ids = []

    for detection in output:
        if detection.shape[0] == 84:
            x, y, w, h = detection[:4]
            scores = detection[4:]
            confidence = np.max(scores)
            class_id = np.argmax(scores)
        else:
            x, y, w, h = detection[:4]
            confidence = detection[4]
            scores = detection[5:]
            class_id = np.argmax(scores)

        if confidence > 0.1:
            # Chuyển từ (x_center, y_center, w, h) sang (x1, y1, x2, y2)
            x1 = int((x - w / 2) * image_shape[1])
            y1 = int((y - h / 2) * image_shape[0])
            x2 = int((x + w / 2) * image_shape[1])
            y2 = int((y + h / 2) * image_shape[0])

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indices

# Xử lý từng ảnh
for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không đọc được ảnh: {image_path}")
        continue

    original_shape = image.shape[:2]
    input_tensor = preprocess(image)
    outputs = session.run(None, {input_name: input_tensor})

    boxes, confidences, class_ids, indices = postprocess(outputs, original_shape)

    # Kiểm tra nếu có bounding box nào được phát hiện
    if indices is not None and len(indices) > 0:
        indices = indices.flatten()

        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            class_id = class_ids[i]
            color = get_color(class_id)

            # Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Vẽ nhãn
            label = f"Class {class_id} {confidences[i]:.2f}"
            cv2.putText(image, label, (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Lưu ảnh
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

print("Hoàn thành! Kết quả đã lưu vào thư mục 'tmp'.")