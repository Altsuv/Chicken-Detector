from ultralytics import YOLO
model = YOLO("yolo26m.pt")
results = model.train(data="Chicken Detector/chicken.yaml", epochs=100, imgsz=640, batch=8, workers=2, device="cuda", close_mosaic=10)
