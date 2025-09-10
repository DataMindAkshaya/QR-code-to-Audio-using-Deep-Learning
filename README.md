# QR Code to Audio for Visually Impaired using Deep Learning

## 📌 Overview
This project converts **QR Codes into audio output** to assist **visually impaired people**.  
It uses a **YOLOv3-tiny deep learning model** to detect and decode QR codes from images or video streams, then converts the extracted data into **speech** using the `pyttsx3` text-to-speech library.

---

## 🚀 Features
- Detects QR codes using **YOLOv3-tiny** object detection.  
- Decodes QR data in real-time.  
- Converts decoded text into **audio output**.  
- Lightweight and fast, suitable for real-time use.  

---

## 🛠️ Tech Stack
- **Python**  
- **YOLOv3-tiny (Darknet/CFG + Weights)**  
- **OpenCV** – image/video processing  
- **NumPy** – matrix operations  
- **Torch & TorchVision** – deep learning backend  
- **pyttsx3** – text-to-speech conversion  

---

## 📂 Project Structure
```
qr-audio-project/
│── qr_test_yolo.py           # Main script
│── qrcode-yolov3-tiny.cfg    # YOLOv3-tiny configuration file
│── qrcode.names              # Class labels file
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
```

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DataMindAkshaya/QR-code-to-Audio-using-Deep-Learning.git
   cd QR-code-to-Audio-using-Deep-Learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pretrained weights:  
   👉 [YOLOv3-tiny QR Code Weights (Google Drive link)]https://drive.google.com/file/d/1YKPYJrc8VhU5l2lmmlEHv0SmZkg_GE1L/view?usp=drive_link

4. Place the weights file in the project folder.  

---

## ▶️ Usage
Run the main script:
```bash
python qr_test_yolo.py
```

- The script will open your webcam or test images.  
- Detected QR codes will be **decoded** and read aloud using **speech output**.  

---

## 📊 Dataset
- Training dataset used: *(provide source, e.g., Kaggle/Custom dataset)*  
- For large datasets, provide an external link (Google Drive / Kaggle).  
- Dataset is **not uploaded to GitHub** due to size constraints.  

---

## 🎧 Example Output
- Input: QR code image with text `"Hello World"`  
- Output:  
  ```
  Detected QR Code: Hello World
  ```
  *(Audio: “Hello World” played through speakers)*  

---

## 📜 License
This project is licensed under the **MIT License** – see the LICENSE file for details.  

---

## 🤝 Contributors
- **Akshaya M** – Developer  
- Supervisor/Trainer (if applicable)  

---

## 🌟 Acknowledgements
- YOLOv3 model by Joseph Redmon  
- OpenCV community  
- pyttsx3 developers  
