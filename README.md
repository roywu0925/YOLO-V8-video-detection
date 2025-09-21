#  README – YOLOv8 Real-Time People Detection

##  Project Overview

This project demonstrates **real-time people detection** in videos using the **YOLOv8n** pretrained model. A video input is processed frame by frame, detecting persons (class ID = 0) and drawing bounding boxes with confidence scores. The project leverages **Ultralytics YOLOv8**, **OpenCV**, and **NumPy**.

Use case: Surveillance, crowd monitoring, or any application where identifying humans in video streams is critical.

---

##  Repository Contents

* **`src/people_movie/main.py`** → Main script for people detection
* **`src/people_movie/__init__.py`** → Package init file
* **`yolov8n.pt`** → Pretrained YOLOv8n model weights
* **`test.mp4`** → Sample video for testing
* **`pyproject.toml`** / **`poetry.lock`** → Poetry environment configuration
* **`tests/`** → Unit test folder (currently empty)

---

##  Key Concepts

* **Model**: YOLOv8n (Ultralytics) pretrained on COCO dataset
* **Target Class**: `person` (class ID = 0)
* **Frameworks**: Ultralytics, OpenCV, NumPy
* **Detection Flow**:

  1. Load pretrained YOLOv8n model
  2. Read video input frame by frame
  3. Run detection → filter for `person` class
  4. Draw bounding boxes + confidence scores
  5. Display results in real-time

---

## 🛠 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/YOLO-V8-video-detection.git
cd YOLO-V8-video-detection
```

### 2. Install Dependencies (via Poetry)

```bash
poetry install
```

### 3. Run Detection

```bash
poetry run python src/people_movie/main.py
```

Press **Q** to exit video playback.

---

##  Key Takeaways

1. YOLOv8n is lightweight and effective for real-time applications
2. OpenCV enables smooth video capture and annotation
3. Bounding boxes + confidence scores improve interpretability

---

##  References

* [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
* [OpenCV Documentation](https://docs.opencv.org)

---

##  Author

* **Roy Wu**
  Statistics + Computer Science + Math background
  Focus: AI systems, computer vision, and deployment
