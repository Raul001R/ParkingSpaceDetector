import cv2
from ultralytics import YOLO
import numpy as np

CAR_CLASS_ID = 2
CONFIDENCE_THRESHOLD = 0.3


def load_model(weights_path="yolov8n.pt"):
    return YOLO(weights_path)


def get_parking_spots():
    #define parking spaces as polylines
    return np.array([[[484,1123],[770,1114],[600,1203],[259,1214]],
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


def open_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        print("Video has opened successfully!")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {frame_count},FPS: {fps}")

    return cap


def draw_car_boxes(frame, boxes):
    #draw bounding boxes and labels for detected cars, return count
    counter = 0
    for box in boxes:
        if int(box.cls[0]) == CAR_CLASS_ID and box.conf[0] > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Car {counter} / confidence {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            counter += 1
    print(f"Detected {counter} cars")
    return counter


def spot_center(spot):
    cx = int((spot[0][0] + spot[1][0] + spot[2][0] + spot[3][0]) / 4)
    cy = int((spot[0][1] + spot[1][1] + spot[2][1] + spot[3][1]) / 4)
    return (cx, cy)


def is_spot_occupied(center, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x1 <= center[0] < x2 and y1 <= center[1] <= y2 and int(box.cls[0]) == CAR_CLASS_ID:
            return True
    return False


def draw_spot(frame, spot, center, occupied):
    if occupied:
        cv2.polylines(frame, [spot], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, "Occupied", center, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        cv2.polylines(frame, [spot], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, "Open", center, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def update_occupancy(frame, spot_corners, boxes, previous_status):
    current_status = {}
    for i, spot in enumerate(spot_corners):
        center = spot_center(spot)
        occupied = is_spot_occupied(center, boxes)

        draw_spot(frame, spot, center, occupied)

        current_status[i] = occupied
        if current_status[i] != previous_status.get(i):
            previous_status[i] = current_status[i]

    return previous_status


def process_frame(model, frame, spot_corners, previous_status):
    #yolo model predicts off of each frame
    results = model.predict(frame)
    boxes = results[0].boxes

    draw_car_boxes(frame, boxes)
    return update_occupancy(frame, spot_corners, boxes, previous_status)


def main():
    model = load_model()
    spot_corners = get_parking_spots()
    previous_status = {}

    cap = open_capture("placeholder")

    #setting up infinite loop that will break whenever a frame isnt read i.e video is over.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        previous_status = process_frame(model, frame, spot_corners, previous_status)

        #write the output frame with parking space occupancy
        cv2.imshow("output_with_parking_spaces.jpg", frame)

        #wait for 1ms for key press to continue or exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #release cap tools and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
