from ultralytics import YOLO

def train_helmet_model():
    # Create a new YOLO model
    model = YOLO('yolov5s.pt')

    # Train the model
    results = model.train(
        data='helmet_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='helmet_model'
    )
    
    # Save the model
    model.save('helmet_model.pt')

if __name__ == "__main__":
    train_helmet_model() 