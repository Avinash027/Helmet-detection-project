import cv2
import torch
import numpy as np
from PIL import Image
import requests
import os
from torchvision import transforms

def download_model():
    # Download YOLOv5 model weights if not exists
    if not os.path.exists('helmet_model.pt'):
        url = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
        response = requests.get(url)
        with open('helmet_model.pt', 'wb') as f:
            f.write(response.content)

def get_color_name(rgb):
    # Define color ranges for white, yellow, and black
    colors = {
        'white': ([200, 200, 200], [255, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'black': ([0, 0, 0], [50, 50, 50])
    }
    
    # Convert RGB to HSV
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    
    for color_name, (lower, upper) in colors.items():
        lower = np.array(lower)
        upper = np.array(upper)
        if np.all(hsv >= lower) and np.all(hsv <= upper):
            return color_name
    return "other"

class HelmetDetector:
    def __init__(self):
        # Download and load YOLOv5 model for both person and helmet detection
        download_model()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Using standard YOLO model
        self.model.conf = 0.5  # Confidence threshold
        
    def detect_helmets(self, frame):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = self.model(rgb_frame)
        
        # Get all person detections
        person_detections = []
        helmet_detections = []
        
        for det in results.xyxy[0]:
            class_id = int(det[-1])
            if class_id == 0:  # person class in COCO dataset
                person_detections.append(det[:4].cpu().numpy())
            elif class_id == 44:  # helmet class (you might need to adjust this class ID)
                helmet_detections.append(det[:4].cpu().numpy())
        
        # Process each person detection
        for person_box in person_detections:
            x1, y1, x2, y2 = map(int, person_box)
            head_region = [x1, y1, x2, int(y1 + (y2-y1)*0.2)]  # Approximate head region
            
            # Check if there's a helmet in the head region
            wearing_helmet = False
            for helmet_box in helmet_detections:
                hx1, hy1, hx2, hy2 = map(int, helmet_box)
                # Check overlap with head region
                if self._boxes_overlap(head_region, [hx1, hy1, hx2, hy2]):
                    wearing_helmet = True
                    # Get helmet color
                    helmet_roi = rgb_frame[hy1:hy2, hx1:hx2]
                    if helmet_roi.size > 0:
                        avg_color = np.mean(helmet_roi, axis=(0,1))
                        color_name = get_color_name(avg_color)
                        
                        # Draw green box for helmet detection
                        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                        label = f"Helmet ({color_name})"
                        cv2.putText(frame, label, (hx1, hy1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break
            
            # Draw red box and "Not Wearing" for people without helmets
            if not wearing_helmet:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Not Wearing", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def _boxes_overlap(self, box1, box2):
        # Check if two boxes overlap
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)

def main():
    # Initialize detector
    detector = HelmetDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = detector.detect_helmets(frame)
        
        # Display result
        cv2.imshow('Helmet Detection', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 