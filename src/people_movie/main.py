from ultralytics import YOLO
import cv2

# 載入模型（使用 yolov8n 預訓練）
model = YOLO("yolov8n.pt")

# 載入影片
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ 無法開啟影片 {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 偵測畫面中的人物
    results = model(frame)

    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # class 0 == person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                label = f"Person: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 - People Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()