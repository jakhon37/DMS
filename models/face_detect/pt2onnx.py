from ultralytics import YOLO

model_path = 'weights/yolov8n-face.pt'
model = YOLO(model_path) 
model.export(format="onnx", imgsz=[480,640])