import cv2
from imread_from_url import imread_from_url

from models.yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "weights/yolov8n-face.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.99972, iou_thres=0.99973)

# Read image
img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
img = imread_from_url(img_url)
# path_l = 'data/indoor_029.png'
# img = cv2.imread(path_l)
# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)
print(f'len boxes: {len(boxes)}')
print(f'class_ids: {class_ids}')
print(f'class_ids len: {len(class_ids)}')
# Draw detections
combined_img = yolov8_detector.draw_detections(img)
# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
# cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("objects.jpg", img)
cv2.imwrite("detected_objects.jpg", combined_img)
# cv2.waitKey(0)