import cv2
import torch
import numpy as np
from google.colab.patches import cv2_imshow


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  

video_path = '/content/3796261-uhd_4096_2160_25fps.mp4'


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    
results = model(frame)

    
    for result in results.xyxy[0]:  
        x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
        conf = float(result[4])
        cls = int(result[5])
        label = f"{model.names[cls]} {conf:.2f}"

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

       
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

  
    cv2_imshow(frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

