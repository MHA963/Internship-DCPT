from ultralytics import YOLO
import time


def train_yolo():
    model = YOLO('yolov8x.pt')
    results = model.train(data='custom_data.yaml', epochs=200, imgsz=640, device=1)

if __name__ == '__main__':
    start_time = time.time()

    train_yolo()
    
    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"Training completed in {hours}:{minutes}:{seconds}.")


## 5 hours and 20 min 