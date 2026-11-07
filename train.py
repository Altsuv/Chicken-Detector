from ultralytics import YOLO
model = YOLO("yolo26m.pt")
results = model.train(data="Chicken Detector/chicken.yaml", epochs=25, imgsz=640, batch=4, workers=0, device="cpu", cache=False, close_mosaic=10)