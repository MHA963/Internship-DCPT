from ultralytics import YOLO

def train_yolo():
    model = YOLO('yolov8x-cls.pt')
    results = model.train(data='E:/Jasper/DBCGRT/MHA/Classifier/dataset', epochs=100, imgsz=640, device=1)

if __name__ == '__main__':
    train_yolo()