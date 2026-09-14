import cv2
from ultralytics import YOLO 
import numpy as np 

#load model and predict image
model = YOLO("yolov8n.pt")
results = model.predict("lot.jpg")

# Load image
img = cv2.imread("lot.jpg")

# Draw bounding boxes and labels on the image
counter = 0
for box in results[0].boxes:
   if int(box.cls[0])== 2 and box.conf[0] > 0.3: #class 2 is car
    x1,y1,x2,y2 = map(int,box.xyxy[0])
    cv2.rectangle (img,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.putText(img,f"Car {counter} / confidence {box.conf[0]:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
    counter += 1  
print(f"Detected {counter} cars")
cv2.imwrite("output.jpg", img) # Save the output image 


#define parking spaces as polylines
spot_corners = np.array([[[484,1123],[770,1114],[600,1203],[259,1214]], 
                         [[876,1108],[1109,1088],[1031,1192],[761,1201]],
                         [[1109,1088],[1328,1100],[1292,1186],[1028,1191]],
                          [[1325,1100],[1544,1094],[1558,1175],[1292,1188]],
                          [[1543,1099],[1761,1088],[1823,1166],[1560,1175]],
                          [[1761,1086],[1983,1071],[2096,1163],[1817,1165]],
                          [[1987,1074],[2202,1074],[2378,1158],[2102,1165]],
                          [[2205,1070],[2490,1056],[2656,1132],[2378,1158]],
                          [[2079,1330],[2463,1336],[2875,1557],[2374,1644]],
                          [[1615,1359],[2067,1336],[2378,1638],[1724,1646]],
                          [[1198,1378],[1615,1358],[1712,1646],[1074,1664]]
                        ], dtype=np.int32)


for spot in spot_corners:
  occupied = False  

  cx = int((spot[0][0]+spot[1][0]+spot[2][0]+spot[3][0]) / 4)
  cy = int((spot[0][1]+spot[1][1]+spot[2][1]+spot[3][1]) / 4)
  center = (cx,cy)

  

  for box in results[0].boxes :
    x1,y1,x2,y2 = map(int,box.xyxy[0])
    if x1 <= center[0] <x2 and y1 <= center[1] <= y2 and int(box.cls[0])== 2:
      occupied = True

  if occupied == True:
    cv2.polylines(img, [spot], isClosed=True, color=(0, 0, 225), thickness=2)
    cv2.putText(img,"Occupied",center,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
  elif occupied == False:
    cv2.polylines(img, [spot], isClosed=True, color=(0, 225, 0), thickness=2)
    cv2.putText(img,"Open",center,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)



#write the output image with parking space occupancy
cv2.imwrite("output_with_parking_spaces.jpg", img)

