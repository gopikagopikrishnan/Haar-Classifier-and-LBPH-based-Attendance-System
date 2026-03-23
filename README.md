# Face Recognition Based Attendance Monitoring System

A Tkinter desktop application that automates student attendance using real-time face detection (Haar cascades) and face recognition (LBPH algorithm via OpenCV).

---

## Features

| Feature | Details |
|---|---|
| **Face capture** | Captures up to 100 face samples per student via webcam |
| **LBPH training** | Trains a Local Binary Patterns Histograms recogniser on captured images |
| **Real-time recognition** | Matches live webcam feed against trained profiles; marks unknown if confidence < 50 |
| **Attendance logging** | Saves daily attendance to a date-stamped CSV (`Attendance/Attendance_DD-MM-YYYY.csv`) |
| **GUI treeview** | Displays today's attendance log inside the app in real time |
| **Live clock** | Shows current time, updated every 200 ms |

---

## Project Structure

```
face-attendance-system/
│
├── main.py                          # Entry point — GUI + all pipeline functions
├── requirements.txt                 # Python dependencies
├── .gitignore
├── README.md
│
├── haarcascade_frontalface_default.xml   # Download separately (see Setup)
│
# Created automatically at runtime:
├── StudentDetails/
│   └── StudentDetails.csv           # Serial No., ID, Name per student
├── TrainingImage/
│   └── <Name>.<Serial>.<ID>.<n>.jpg # Captured face samples
├── TrainingImageLabel/
│   └── Trainner.yml                 # Saved LBPH model
└── Attendance/
    └── Attendance_DD-MM-YYYY.csv    # Daily attendance log
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/face-attendance-system.git
cd face-attendance-system
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Haar cascade file

Download `haarcascade_frontalface_default.xml` from the OpenCV GitHub repository and place it in the **project root**:

```
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
```

Or run:

```bash
curl -L -o haarcascade_frontalface_default.xml \
  https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

### 5. Run the application

```bash
python main.py
```

---

## Usage

### Registering a new student

1. Enter the student **ID** and **Name** in the right panel.
2. Click **Take Images** — the webcam opens and captures up to 100 face samples.  
   Press **Q** to stop early.
3. Click **Save Profile** — trains the LBPH model on all captured images.

### Taking attendance

1. Click **Take Attendance** in the left panel — the webcam opens.
2. Recognised faces are matched against the trained model.  
   - Confidence **< 50** → student is identified and attendance is logged.  
   - Confidence **≥ 50** → face is marked as *Unknown*.
3. Press **Q** to stop.  
   Attendance is written to `Attendance/Attendance_<date>.csv` and displayed in the treeview.

---

## Key Functions

| Function | Purpose |
|---|---|
| `assure_path_exists(path)` | Creates a directory if it does not exist |
| `check_haarcascadefile()` | Validates presence of the Haar XML; closes app if missing |
| `tick()` | Recursive 200 ms callback to update the live clock label |
| `TakeImages()` | Webcam capture → saves grayscale face crops to `TrainingImage/` |
| `getImagesAndLabels(path)` | Reads all images in a folder → returns `(faces, ids)` for training |
| `TrainImages()` | Trains LBPH recogniser and saves `Trainner.yml` |
| `TrackImages()` | Real-time recognition → logs attendance CSV + updates treeview |

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Haar cascade detection |
| `opencv-contrib-python` | LBPH face recogniser (`cv2.face`) |
| `numpy` | Array operations |
| `Pillow` | Image loading and grayscale conversion |
| `pandas` | Reading student details CSV |

---

## Notes

- The Haar cascade XML is **not** committed to the repository (see `.gitignore`). Download it as shown above.
- Training images and attendance logs are also excluded from version control. Add appropriate cloud backup if needed.
- Python **3.8+** recommended.
- Tkinter is included with most standard Python distributions. If missing: `sudo apt-get install python3-tk` (Linux).

---

## License

MIT
