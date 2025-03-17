import numpy as np
import cv2
from onnxruntime import ort

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

class BoundingBox:
    def __init__(self, *res):
        self.label = res[0]
        self.prob = res[1]
        self.x1 = res[2]
        self.y1 = res[3]
        self.x2 = res[4]
        self.y2 = res[5]
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.cen_x = int((self.x1+self.x2) / 2)
        self.cen_y = int((self.y1+self.y2)/2)
        self.area = self.w * self.h
    def display(self):
        print("label: ", self.label, "cenx: ", self.cen_x, "ceny: ", self.cen_y)

class Yolov10Model():
    def __init__(self, model_path, img_size=640, device='cpu'):
        self.device = device
        self.session = ort.InferenceSession(model_path)
        self.img_size = (img_size, img_size)

    def predict(self, img):
        print("READ IMAGE SUCCESSFULLY")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, C = image.shape
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        session = self.session
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]:im}
        outputs = session.run(outname, inp)[0]
        boxes = []
        #for _, (batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        for _, (x0,y0,x1,y1,score,cls_id) in enumerate(outputs[0]):
            #batch_id = int(batch_id)
            score = round(float(score),3)
            if score < 0.2: continue
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            box = BoundingBox(cls_id, score, max(0,box[0]), max(0,box[1]), min(W,box[2]), min(H,box[3]))
            boxes.append(box)
        return boxes