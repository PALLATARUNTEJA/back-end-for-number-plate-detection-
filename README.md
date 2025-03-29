# Number Plate Detection System (Backend)

A **Flask-based backend** for number plate detection using **YOLOv8** and **PaddleOCR**. This system processes images to detect and extract number plates, with an optional image dehazing feature for improved recognition.

## 🚀 Features
- REST API for number plate detection
- Two processing modes:
  - Direct number plate extraction
  - Image dehazing + number plate extraction
- Fast and efficient inference using YOLOv8 and PaddleOCR
- Supports batch image processing

## 💻 Technologies Used
- **Backend**: Flask, FastAPI (Optional), Python
- **Machine Learning**: YOLOv8, PaddleOCR, OpenCV, NumPy, PyTorch
- **Dehazing**: Dark Channel Prior (DCP) Algorithm

## ⚙️ How to Run the Project
1. Clone the backend repository:
   ```bash
   git clone https://github.com/PALLATARUNTEJA/back-end-for-number-plate-detection-.git
   ```
2. Navigate to the project directory:
   ```bash
   cd back-end-for-number-plate-detection-
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Flask server:
   ```bash
   python app.py
   ```
6. The API will be available at `http://127.0.0.1:5000` by default.

## 📂 Project Structure
```
back-end-for-number-plate-detection-/
├── models/              # YOLOv8 and PaddleOCR models
├── static/uploads/      # Uploaded images
├── utils/               # Helper functions (e.g., image processing, OCR handling)
├── app.py               # Main Flask application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

## 🚀 API Endpoints
- `POST /detect` - Upload an image for number plate detection
- `POST /detect?mode=dehaze` - Upload an image with dehazing applied before detection
- `GET /status` - Check if the server is running

## 🚀 Future Enhancements
- Add support for video input processing
- Deploy the model on cloud (AWS, GCP, or Azure)
- Implement a database to store detected plates

## 🙌 Contributing
Contributions are welcome! Feel free to fork the project, make changes, and submit a pull request.

## 📜 License
This project is licensed to **PALLATARUNTEJA**.

