"""
Face Recognition Based Attendance Monitoring System
====================================================
A Tkinter GUI application that uses OpenCV Haar cascades and LBPH
face recognition to automate student attendance tracking.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def assure_path_exists(path):
    """Create directory (and any intermediate dirs) if it does not exist."""
    dir_ = os.path.dirname(path)
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def check_haarcascadefile():
    """Verify that the Haar cascade XML file is present; close app if not."""
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess.showwarning(
            title="File Missing",
            message="haarcascade_frontalface_default.xml not found.\n"
                    "Please contact us for help."
        )
        window.destroy()


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

def tick():
    """Update the clock label every 200 ms (recursive callback)."""
    time_string = time.strftime("%H:%M:%S")
    clock.config(text=time_string)
    clock.after(200, tick)


# ---------------------------------------------------------------------------
# GUI field helpers
# ---------------------------------------------------------------------------

def clear():
    """Clear the ID entry field and reset the status message."""
    txt.delete(0, "end")
    res = "1) Take Images  >>>  2) Save Profile"
    message1.configure(text=res)


def clear2():
    """Clear the Name entry field and reset the status message."""
    txt2.delete(0, "end")
    res = "1) Take Images  >>>  2) Save Profile"
    message1.configure(text=res)


# ---------------------------------------------------------------------------
# Core face-recognition pipeline
# ---------------------------------------------------------------------------

def TakeImages():
    """
    Capture up to 100 face images for a new student and store them in
    TrainingImage/.  Student details are appended to StudentDetails.csv.
    """
    check_haarcascadefile()

    columns = ["SERIAL NO.", "", "ID", "", "NAME"]
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    # Determine next serial number
    serial = 0
    csv_path = os.path.join("StudentDetails", "StudentDetails.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for _ in reader:
                serial += 1
        serial = serial // 2
    else:
        with open(csv_path, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            serial = 1

    # Read user input
    Id = txt.get()
    name = txt2.get()

    if name.isalpha() or " " in name:
        cam = cv2.VideoCapture(0)
        harcascade_path = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascade_path)
        sample_num = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sample_num += 1

                # Save the cropped face region
                face_filename = (
                    f"TrainingImage/{name}.{str(serial)}.{str(Id)}.{str(sample_num)}.jpg"
                )
                cv2.imwrite(face_filename, gray[y: y + h, x: x + w])
                cv2.imshow("Taking Images", img)

            # Stop on 'q' or after 100 samples
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            elif sample_num > 100:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save student details to CSV
        res = f"Images Taken for ID : {Id}"
        row = [serial, "", Id, "", name]
        with open(csv_path, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        message1.configure(text=res)

    else:
        if not name.isalpha():
            res = "Enter a Correct Name (letters only)"
            message.configure(text=res)


def getImagesAndLabels(path):
    """
    Load grayscale face images and their integer ID labels from *path*.

    Returns
    -------
    faces : list of np.ndarray  (uint8)
    ids   : list of int
    """
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for image_path in image_paths:
        # Convert to grayscale PIL image → NumPy array
        pil_image = Image.open(image_path).convert("L")
        image_np = np.array(pil_image, "uint8")

        # Extract numeric ID encoded in the filename (e.g. Name.serial.ID.sample.jpg)
        id_ = int(os.path.split(image_path)[-1].split(".")[2])

        faces.append(image_np)
        ids.append(id_)

    return faces, ids


def TrainImages():
    """
    Train an LBPH face recognizer on the images stored in TrainingImage/
    and save the model to TrainingImageLabel/Trainner.yml.
    """
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascade_path = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascade_path)

    faces, ids = getImagesAndLabels("TrainingImage")

    try:
        recognizer.train(faces, np.array(ids))
    except Exception:
        mess.showwarning(
            title="No Registrations",
            message="Please register at least one student first."
        )
        return

    recognizer.save(os.path.join("TrainingImageLabel", "Trainner.yml"))
    res = "Profile Saved Successfully"
    message1.configure(text=res)
    message.configure(text=f"Total Registrations till now : {str(ids[0])}")


def TrackImages():
    """
    Use the trained LBPH model to recognise faces from the webcam in
    real-time and log attendance to a date-stamped CSV in Attendance/.
    """
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")

    # Clear existing treeview rows
    for row in tv.get_children():
        tv.delete(row)

    msg = ""
    i = 0
    j = 0

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = os.path.join("TrainingImageLabel", "Trainner.yml")

    if os.path.isfile(model_path):
        recognizer.read(model_path)
    else:
        mess.showwarning(
            title="Data Missing",
            message="Please click on Save Profile to reset data!!"
        )
        return

    harcascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(harcascade_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    col_names = ["Id", "", "Name", "", "Date", "", "Time"]
    student_csv = os.path.join("StudentDetails", "StudentDetails.csv")

    if os.path.isfile(student_csv):
        df = pd.read_csv(student_csv)
    else:
        mess.showwarning(
            title="Details Missing",
            message="Student details are missing, please check!"
        )
        return

    cam = cv2.VideoCapture(0)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y: y + h, x: x + w])

            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")

                aa = df.loc[df["SERIAL NO."] == serial]["NAME"].values
                Id = df.loc[df["SERIAL NO."] == serial]["ID"].values
                Id = str(Id)
                Id = Id[1:-1]
                bb = str(aa)
                bb = bb[2:-2]

                attendance = [str(Id), "", bb, "", str(date), "", str(timestamp)]
            else:
                Id = "Unknown"
                bb = str(Id)
                attendance = []

            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)

        cv2.imshow("Taking Attendance", im)

        if cv2.waitKey(1) == ord("q"):
            break

    # Determine attendance CSV path
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    attendance_csv = os.path.join("Attendance", f"Attendance_{date}.csv")

    if os.path.isfile(attendance_csv):
        with open(attendance_csv, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(attendance)
    else:
        with open(attendance_csv, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            writer.writerow(attendance)

    # Populate treeview from CSV
    with open(attendance_csv, "r") as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            i += 1
            if i > 1:
                if i % 2 != 0:
                    iidd = str(line[0]) + "   "
                    tv.insert("", 0, text=iidd, values=(str(line[2]), str(line[4]), str(line[6])))

    cam.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Global date / time variables
# ---------------------------------------------------------------------------

global key
key = ""

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
day, month, year = date.split("-")

mont = {
    "01": "January", "02": "February", "03": "March",
    "04": "April",   "05": "May",      "06": "June",
    "07": "July",    "08": "August",   "09": "September",
    "10": "October", "11": "November", "12": "December",
}


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("Attendance System")
window.configure(background="#2d420a")

# ── Frames ──────────────────────────────────────────────────────────────────
frame1 = tk.Frame(window, bg="#c79cff")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#c79cff")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

frame3 = tk.Frame(window, bg="#c6cce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c6cce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

# ── Title ───────────────────────────────────────────────────────────────────
message3 = tk.Label(
    window,
    text="Face Recognition Based Attendance Monitoring System",
    fg="white", bg="#2d420a",
    width=55, height=1,
    font=("comic", 22, "bold"),
)
message3.place(x=10, y=10)

# ── Date & Clock ────────────────────────────────────────────────────────────
datef = tk.Label(
    frame4,
    text=f"{day}+'-'+{mont[month]}+'-'+{year}   |   ",
    fg="#ff61e5", bg="#2d420a",
    width=55, height=1,
    font=("comic", 22, "bold"),
)
datef.pack(fill="both", expand=1)

clock = tk.Label(frame3, fg="#ff61e5", bg="#2d420a", width=55, height=1, font=("comic", 22, "bold"))
clock.pack(fill="both", expand=1)
tick()

# ── Section headings ────────────────────────────────────────────────────────
head2 = tk.Label(
    frame2,
    text="                         For New Registrations                        ",
    fg="black", bg="#00fcca",
    font=("comic", 17, "bold"),
)
head2.grid(row=0, column=0)

head1 = tk.Label(
    frame1,
    text="                         For Already Registered                       ",
    fg="black", bg="#00fcca",
    font=("comic", 17, "bold"),
)
head1.place(x=0, y=0)

# ── Input widgets (frame2) ───────────────────────────────────────────────────
lbl = tk.Label(frame2, text="Enter ID", width=20, height=1, fg="black", bg="#c79cff", font=("comic", 17, "bold"))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=("comic", 15, "bold"))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", width=20, height=1, fg="black", bg="#c79cff", font=("comic", 17, "bold"))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=("comic", 15, "bold"))
txt2.place(x=30, y=173)

message1 = tk.Label(
    frame2,
    text="1) Take Images  >>>  2) Save Profile",
    bg="#c79cff", fg="black",
    width=39, height=1,
    activebackground="#3ffc00",
    font=("comic", 16, "bold"),
)
message1.place(x=7, y=230)

message = tk.Label(
    frame2,
    text="",
    bg="#c79cff", fg="black",
    width=39, height=1,
    activebackground="#3ffc00",
    font=("comic", 16, "bold"),
)
message.place(x=7, y=450)

# ── Attendance label (frame1) ────────────────────────────────────────────────
lbl3 = tk.Label(frame1, text="Attendance", width=20, height=1, fg="black", bg="#c79cff", font=("comic", 17, "bold"))
lbl3.place(x=100, y=115)

# ── Total registrations count ────────────────────────────────────────────────
res = 0
student_csv = os.path.join("StudentDetails", "StudentDetails.csv")
if os.path.isfile(student_csv):
    with open(student_csv, "r") as f:
        reader = csv.reader(f)
        for _ in reader:
            res += 1
    res = (res // 2) - 1
else:
    res = 0
message.configure(text=f"Total Registrations till now : {str(res)}")

# ── Menu bar ─────────────────────────────────────────────────────────────────
menubar = tk.Menu(window, relief="ridge")
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Exit", command=window.destroy)
menubar.add_cascade(label="Help", font=("comic", 29, "bold"), menu=filemenu)

# ── Attendance Treeview (frame1) ─────────────────────────────────────────────
tv = ttk.Treeview(frame1, height=13, columns=("name", "date", "time"))
tv.column("#0",    width=82)
tv.column("name",  width=130)
tv.column("date",  width=133)
tv.column("time",  width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)

tv.heading("#0",   text="ID")
tv.heading("name", text="NAME")
tv.heading("date", text="DATE")
tv.heading("time", text="TIME")

scroll = ttk.Scrollbar(frame1, orient="vertical", command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky="ns")
tv.configure(yscrollcommand=scroll.set)

# ── Buttons ───────────────────────────────────────────────────────────────────
clearButton = tk.Button(
    frame2, text="Clear", command=clear,
    fg="black", bg="#ff7221", width=11,
    activebackground="white", font=("comic", 15, "bold"),
)
clearButton.place(x=335, y=86)

clearButton2 = tk.Button(
    frame2, text="Clear", command=clear2,
    fg="black", bg="#ff7221", width=11,
    activebackground="white", font=("comic", 15, "bold"),
)
clearButton2.place(x=335, y=172)

takeImg = tk.Button(
    frame2, text="Take Images", command=TakeImages,
    fg="white", bg="#6d00fc", width=34, height=1,
    activebackground="white", font=("comic", 15, "bold"),
)
takeImg.place(x=30, y=300)

trainImg = tk.Button(
    frame2, text="Save Profile", command=TrainImages,
    fg="white", bg="#6d00fc", width=34, height=1,
    activebackground="white", font=("comic", 15, "bold"),
)
trainImg.place(x=30, y=380)

trackImg = tk.Button(
    frame1, text="Take Attendance", command=TrackImages,
    fg="black", bg="#3ffc00", width=35, height=1,
    activebackground="white", font=("comic", 15, "bold"),
)
trackImg.place(x=30, y=50)

quitWindow = tk.Button(
    frame1, text="Quit", command=window.destroy,
    fg="black", bg="#eb4600", width=35, height=1,
    activebackground="white", font=("comic", 15, "bold"),
)
quitWindow.place(x=30, y=450)

# ── Start event loop ─────────────────────────────────────────────────────────
window.configure(menu=menubar)
window.mainloop()
