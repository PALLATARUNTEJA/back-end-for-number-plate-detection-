
# from flask import Flask, request, jsonify, send_from_directory
# import cv2
# import numpy as np
# import os
# import re
# import torch
# from paddleocr import PaddleOCR
# from werkzeug.utils import secure_filename
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Load OWLv2 model (replace with your OWLv2 implementation)
# def load_owlv2_model():
#     # Example: Load a pre-trained object detection model
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Replace with OWLv2
#     return model

# owlv2_model = load_owlv2_model()

# # Ensure upload and result directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# # Dark Channel Prior for dehazing
# def dark_channel_prior(image, window_size=15):
#     """Compute the dark channel prior of an image."""
#     min_channel = np.min(image, axis=2)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
#     dark_channel = cv2.erode(min_channel, kernel)
#     return dark_channel

# def estimate_atmospheric_light(image, dark_channel):
#     """Estimate the atmospheric light from the dark channel."""
#     flat_image = image.reshape(-1, 3)
#     flat_dark = dark_channel.ravel()
#     top_percent = 0.001  # Top 0.1% brightest pixels
#     num_pixels = int(flat_dark.shape[0] * top_percent)
#     indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
#     atmospheric_light = np.max(flat_image[indices], axis=0)
#     return atmospheric_light

# def dehaze_image(image, window_size=15, omega=0.95, t0=0.1):
#     """Dehaze an image using the Dark Channel Prior."""
#     # Convert image to float32 for calculations
#     image = image.astype(np.float32) / 255.0

#     # Compute dark channel and atmospheric light
#     dark_channel = dark_channel_prior(image, window_size)
#     atmospheric_light = estimate_atmospheric_light(image, dark_channel)

#     # Compute transmission map
#     transmission = 1 - omega * dark_channel_prior(image / atmospheric_light, window_size)
#     transmission = np.clip(transmission, t0, 1.0)

#     # Recover the scene radiance
#     dehazed_image = np.zeros_like(image)
#     for i in range(3):
#         dehazed_image[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]

#     # Clip and convert back to uint8
#     dehazed_image = np.clip(dehazed_image, 0, 1)
#     dehazed_image = (dehazed_image * 255).astype(np.uint8)

#     return dehazed_image

# # Detect number plate region using OWLv2
# def detect_number_plate_region(image):
#     """Detect the number plate region using OWLv2."""
#     results = owlv2_model(image)  # Replace with OWLv2 inference
#     plates = results.xyxy[0].cpu().numpy()  # Get bounding boxes
#     if len(plates) > 0:
#         x1, y1, x2, y2, conf, cls = plates[0]  # Get the first detected plate
#         return int(x1), int(y1), int(x2), int(y2)
#     return None

# # Number plate detection and OCR
# def detect_number_plate(image_path):
#     image = cv2.imread(image_path)
#     plate_region = detect_number_plate_region(image)
#     if plate_region:
#         x1, y1, x2, y2 = plate_region
#         cropped_plate = image[y1:y2, x1:x2]
#         result = ocr.ocr(cropped_plate)
#         text = ""
#         for line in result:
#             for word in line:
#                 text += word[1][0] + " "

#         # Filter out non-alphanumeric characters and spaces
#         filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
#         return filtered_text
#     return "No number plate detected"

# @app.route('/')
# def index():
#     return "Flask Backend for Number Plate Detection"

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     option = request.form.get('option')

#     if option == '1':
#         # Option 1: Extract number plate
#         number_plate_text = detect_number_plate(file_path)
#         return jsonify({
#             "result": number_plate_text,
#             "image": filename,
#             "dehazed_image": None
#         })

#     elif option == '2':
#         # Option 2: Dehaze image and extract number plate
#         image = cv2.imread(file_path)
#         dehazed_image = dehaze_image(image)
#         dehazed_path = os.path.join(app.config['RESULT_FOLDER'], f"dehazed_{filename}")
#         cv2.imwrite(dehazed_path, dehazed_image)

#         number_plate_text = detect_number_plate(dehazed_path)
#         return jsonify({
#             "result": number_plate_text,
#             "image": filename,
#             "dehazed_image": f"dehazed_{filename}"
#         })

#     else:
#         return jsonify({"error": "Invalid option"}), 400

# @app.route('/uploads/<filename>')
# def get_uploaded_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/results/<filename>')
# def get_result_image(filename):
#     return send_from_directory(app.config['RESULT_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)











from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import re
import torch
from paddleocr import PaddleOCR
from werkzeug.utils import secure_filename
from flask_cors import CORS
from ultralytics import YOLO  # YOLOv8 for number plate detection

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# File upload and result directories
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load YOLOv8 model for number plate detection
def load_yolov8_model():
    model = YOLO('yolov8n.pt')  # Replace with a custom YOLOv8 model trained for number plates
    return model

yolov8_model = load_yolov8_model()

# Dark Channel Prior for dehazing 
def dark_channel_prior(image, window_size=15):
    """Compute the dark channel prior of an image."""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    """Estimate the atmospheric light from the dark channel."""
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    top_percent = 0.001  # Top 0.1% brightest pixels
    num_pixels = int(flat_dark.shape[0] * top_percent)
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    atmospheric_light = np.max(flat_image[indices], axis=0)
    return atmospheric_light

def dehaze_image(image, window_size=15, omega=0.95, t0=0.1):
    """Dehaze an image using the Dark Channel Prior."""
    # Convert image to float32 for calculations
    image = image.astype(np.float32) / 255.0

    # Compute dark channel and atmospheric light
    dark_channel = dark_channel_prior(image, window_size)
    atmospheric_light = estimate_atmospheric_light(image, dark_channel)

    # Compute transmission map
    transmission = 1 - omega * dark_channel_prior(image / atmospheric_light, window_size)
    transmission = np.clip(transmission, t0, 1.0)

    # Recover the scene radiance
    dehazed_image = np.zeros_like(image)
    for i in range(3):
        dehazed_image[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]

    # Clip and convert back to uint8
    dehazed_image = np.clip(dehazed_image, 0, 1)
    dehazed_image = (dehazed_image * 255).astype(np.uint8)

    return dehazed_image

# Detect number plate region using YOLOv8
def detect_number_plate_region(image):
    """Detect the number plate region using YOLOv8."""
    results = yolov8_model(image)  # Run inference
    plates = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    if len(plates) > 0:
        x1, y1, x2, y2 = plates[0]  # Get the first detected plate
        return int(x1), int(y1), int(x2), int(y2)
    return None

# Number plate detection and OCR
def detect_number_plate(image_path):
    """Detect and extract text from the number plate."""
    image = cv2.imread(image_path)
    plate_region = detect_number_plate_region(image)
    if plate_region:
        x1, y1, x2, y2 = plate_region
        cropped_plate = image[y1:y2, x1:x2]
        result = ocr.ocr(cropped_plate)
        text = ""
        for line in result:
            for word in line:
                text += word[1][0] + " "

        # Filter out non-alphanumeric characters and spaces
        filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
        return filtered_text
    return "No number plate detected"

# Flask Routes
@app.route('/')
def index():
    return "Flask Backend for Number Plate Detection"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    option = request.form.get('option')

    if option == '1':
        # Option 1: Extract number plate
        number_plate_text = detect_number_plate(file_path)
        return jsonify({
            "result": number_plate_text,
            "image": filename,
            "dehazed_image": None
        })

    elif option == '2':
        # Option 2: Dehaze image and extract number plate
        image = cv2.imread(file_path)
        dehazed_image = dehaze_image(image)
        dehazed_path = os.path.join(app.config['RESULT_FOLDER'], f"dehazed_{filename}")
        cv2.imwrite(dehazed_path, dehazed_image)

        number_plate_text = detect_number_plate(dehazed_path)
        return jsonify({
            "result": number_plate_text,
            "image": filename,
            "dehazed_image": f"dehazed_{filename}"
        })

    else:
        return jsonify({"error": "Invalid option"}), 400

@app.route('/uploads/<filename>')
def get_uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def get_result_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)














# from flask import Flask, request, jsonify, send_from_directory
# import numpy as np
# import os
# import re
# import torch
# from paddleocr import PaddleOCR
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# from ultralytics import YOLO  # YOLOv8 for number plate detection
# from skimage import io, exposure  # Scikit-Image for image processing

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # File upload and result directories
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Load YOLOv8 model for number plate detection
# def load_yolov8_model():
#     model = YOLO('yolov8n.pt')  # Replace with a custom YOLOv8 model trained for number plates
#     return model

# yolov8_model = load_yolov8_model()

# # Dehazing using Scikit-Image (example: contrast stretching)
# def dehaze_image(image):
#     """
#     Dehaze an image using contrast stretching.
#     Replace this with a more advanced dehazing algorithm if needed.
#     """
#     # Convert image to float32 for processing
#     image = image.astype(np.float32) / 255.0

#     # Apply contrast stretching
#     p2, p98 = np.percentile(image, (2, 98))
#     dehazed_image = exposure.rescale_intensity(image, in_range=(p2, p98))

#     # Convert back to uint8
#     dehazed_image = (dehazed_image * 255).astype(np.uint8)
#     return dehazed_image

# # Detect number plate region using YOLOv8
# def detect_number_plate_region(image):
#     """Detect the number plate region using YOLOv8."""
#     # Ensure the image has 3 channels (RGB)
#     if image.shape[2] == 4:  # If the image has 4 channels (RGBA)
#         image = image[:, :, :3]  # Remove the alpha channel

#     results = yolov8_model(image)  # Run inference
#     plates = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
#     if len(plates) > 0:
#         x1, y1, x2, y2 = plates[0]  # Get the first detected plate
#         return int(x1), int(y1), int(x2), int(y2)
#     return None

# # Number plate detection and OCR
# def detect_number_plate(image_path):
#     """Detect and extract text from the number plate."""
#     # Load image using Scikit-Image
#     image = io.imread(image_path)
#     plate_region = detect_number_plate_region(image)
#     if plate_region:
#         x1, y1, x2, y2 = plate_region
#         cropped_plate = image[y1:y2, x1:x2]
        
#         # Perform OCR on the cropped plate
#         result = ocr.ocr(cropped_plate)
        
#         # Check if OCR result is not None
#         if result is None:
#             return "No text detected in the number plate region"
        
#         text = ""
#         for line in result:
#             if line is not None:  # Ensure line is not None
#                 for word in line:
#                     if word is not None and len(word) >= 2:  # Ensure word is not None and has at least 2 elements
#                         text += word[1][0] + " "  # Append the detected text

#         # Filter out non-alphanumeric characters and spaces
#         filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
#         return filtered_text if filtered_text else "No text detected in the number plate region"
#     return "No number plate detected"

# # Flask Routes
# @app.route('/')
# def index():
#     return "Flask Backend for Number Plate Detection"

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     option = request.form.get('option')

#     if option == '1':
#         # Option 1: Extract number plate
#         number_plate_text = detect_number_plate(file_path)
#         return jsonify({
#             "result": number_plate_text,
#             "image": filename,
#             "dehazed_image": None
#         })

#     elif option == '2':
#         # Option 2: Dehaze image and extract number plate
#         image = io.imread(file_path)
#         dehazed_image = dehaze_image(image)
#         dehazed_path = os.path.join(app.config['RESULT_FOLDER'], f"dehazed_{filename}")
#         io.imsave(dehazed_path, dehazed_image)

#         number_plate_text = detect_number_plate(dehazed_path)
#         return jsonify({
#             "result": number_plate_text,
#             "image": filename,
#             "dehazed_image": f"dehazed_{filename}"
#         })

#     else:
#         return jsonify({"error": "Invalid option"}), 400

# @app.route('/uploads/<filename>')
# def get_uploaded_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/results/<filename>')
# def get_result_image(filename):
#     return send_from_directory(app.config['RESULT_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)






# from flask import Flask, request, jsonify, send_from_directory
# import cv2
# import numpy as np
# import os
# import re
# import torch
# from paddleocr import PaddleOCR
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# from ultralytics import YOLO  # YOLOv8 for object detection

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# CROPPED_FOLDER = 'cropped'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER
# app.config['CROPPED_FOLDER'] = CROPPED_FOLDER

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Load YOLOv8 model for number plate detection
# def load_yolov8_model():
#     model = YOLO("yolov8n.pt")  # Load YOLOv8 Nano (smallest model)
#     return model

# yolov8_model = load_yolov8_model()

# # Ensure upload, result, and cropped directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
# os.makedirs(CROPPED_FOLDER, exist_ok=True)

# # Improved Dehazing using AOD-Net (Atmospheric Scattering Model)
# def aod_dehaze(image, omega=0.95, t0=0.1):
#     """Dehaze an image using the AOD-Net inspired approach."""
#     # Convert image to float32 for calculations
#     image = image.astype(np.float32) / 255.0

#     # Compute dark channel
#     dark_channel = np.min(image, axis=2)

#     # Estimate atmospheric light
#     top_percent = 0.001  # Top 0.1% brightest pixels
#     num_pixels = int(dark_channel.size * top_percent)
#     indices = np.argpartition(dark_channel.ravel(), -num_pixels)[-num_pixels:]
#     atmospheric_light = np.max(image.reshape(-1, 3)[indices], axis=0)

#     # Compute transmission map
#     transmission = 1 - omega * (dark_channel / atmospheric_light.max())
#     transmission = np.clip(transmission, t0, 1.0)

#     # Recover the scene radiance
#     dehazed_image = np.zeros_like(image)
#     for i in range(3):
#         dehazed_image[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]

#     # Clip and convert back to uint8
#     dehazed_image = np.clip(dehazed_image, 0, 1)
#     dehazed_image = (dehazed_image * 255).astype(np.uint8)

#     return dehazed_image

# # Post-dehazing enhancement
# def enhance_image(image):
#     """Enhance the image after dehazing."""
#     # Convert to LAB color space for contrast enhancement
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     # Apply CLAHE to the L channel
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)

#     # Merge the channels and convert back to BGR
#     lab = cv2.merge((l, a, b))
#     enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#     # Sharpen the image
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     sharpened_image = cv2.filter2D(enhanced_image, -1, kernel)

#     return sharpened_image

# # Preprocess image for better detection
# def preprocess_image(image):
#     # Resize image
#     image = cv2.resize(image, (640, 640))

#     # Enhance contrast
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)
#     lab = cv2.merge((l, a, b))
#     image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#     return image

# # Detect number plate region using YOLOv8
# def detect_number_plate_region(image):
#     """Detect the number plate region using YOLOv8."""
#     # Preprocess image
#     processed_image = preprocess_image(image)

#     # Perform inference with lower confidence threshold
#     results = yolov8_model(processed_image, conf=0.3)  # Adjust confidence threshold
#     plates = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
#     if len(plates) > 0:
#         x1, y1, x2, y2 = plates[0]  # Get the first detected plate
#         return int(x1), int(y1), int(x2), int(y2)
#     return None

# # Filter OCR results to extract the most likely number plate text
# def filter_number_plate_text(text):
#     """Filter OCR results to extract the most likely number plate text."""
#     # Use regex to find alphanumeric sequences that resemble number plates
#     matches = re.findall(r'\b[A-Z0-9]{6,10}\b', text)  # Adjust regex as needed
#     if matches:
#         return matches[0]  # Return the first match
#     return "No valid number plate text detected"

# # Number plate detection and OCR
# def detect_number_plate(image_path):
#     image = cv2.imread(image_path)
#     plate_region = detect_number_plate_region(image)
#     if plate_region:
#         x1, y1, x2, y2 = plate_region
#         cropped_plate = image[y1:y2, x1:x2]

#         # Preprocess the cropped plate (optional: resize, sharpen, etc.)
#         cropped_plate = cv2.resize(cropped_plate, (300, 100))  # Resize for better OCR

#         # Perform OCR on the cropped plate
#         result = ocr.ocr(cropped_plate)
#         text = ""
#         if result and len(result) > 0:  # Check if OCR result is not None or empty
#             for line in result:
#                 if line and len(line) > 0:  # Check if line is not None or empty
#                     for word in line:
#                         if word and len(word) > 1:  # Check if word is not None and has text
#                             text += word[1][0] + " "

#         # Filter out unwanted characters and extract the most likely number plate text
#         filtered_text = filter_number_plate_text(text)
#         return filtered_text
#     else:
#         # Fallback: Detect closely packed characters in the entire image
#         result = ocr.ocr(image)
#         text = ""
#         if result and len(result) > 0:
#             for line in result:
#                 if line and len(line) > 0:
#                     for word in line:
#                         if word and len(word) > 1:
#                             text += word[1][0] + " "

#         # Filter out unwanted characters and extract the most likely number plate text
#         filtered_text = filter_number_plate_text(text)
#         return filtered_text

# @app.route('/')
# def index():
#     return "Flask Backend for Number Plate Detection"

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     # Dehaze the image
#     image = cv2.imread(file_path)
#     dehazed_image = aod_dehaze(image)
#     enhanced_image = enhance_image(dehazed_image)
#     dehazed_path = os.path.join(app.config['RESULT_FOLDER'], f"dehazed_{filename}")
#     cv2.imwrite(dehazed_path, enhanced_image)

#     # Detect number plate
#     number_plate_text = detect_number_plate(dehazed_path)
#     if number_plate_text == "No valid number plate text detected":
#         # If no number plate is detected, allow cropping
#         return jsonify({
#             "result": "No number plate detected. Please crop the number plate manually.",
#             "image": filename,
#             "dehazed_image": f"dehazed_{filename}",
#             "allow_crop": True
#         })
#     else:
#         return jsonify({
#             "result": number_plate_text,
#             "image": filename,
#             "dehazed_image": f"dehazed_{filename}",
#             "allow_crop": False
#         })

# @app.route('/crop', methods=['POST'])
# def crop_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['CROPPED_FOLDER'], filename)
#     file.save(file_path)

#     # Perform OCR on the cropped image
#     result = ocr.ocr(file_path)
#     text = ""
#     if result and len(result) > 0:
#         for line in result:
#             if line and len(line) > 0:
#                 for word in line:
#                     if word and len(word) > 1:
#                         text += word[1][0] + " "

#     # Filter out unwanted characters and extract the most likely number plate text
#     filtered_text = filter_number_plate_text(text)
#     return jsonify({
#         "result": filtered_text if filtered_text else "No valid number plate text detected",
#         "image": filename
#     })

# @app.route('/uploads/<filename>')
# def get_uploaded_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/results/<filename>')
# def get_result_image(filename):
#     return send_from_directory(app.config['RESULT_FOLDER'], filename)

# @app.route('/cropped/<filename>')
# def get_cropped_image(filename):
#     return send_from_directory(app.config['CROPPED_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)











# from flask import Flask, request, jsonify, send_from_directory
# import cv2
# import numpy as np
# import os
# import re
# import torch
# from paddleocr import PaddleOCR
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# from ultralytics import YOLO  # YOLOv8 for object detection

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Load YOLOv8 model for number plate detection
# def load_yolov8_model():
#     model = YOLO("yolov8n.pt")  # Load YOLOv8 Nano (smallest model)
#     return model

# yolov8_model = load_yolov8_model()

# # Ensure upload and result directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# # Improved Dehazing using AOD-Net (Atmospheric Scattering Model)
# def aod_dehaze(image, omega=0.95, t0=0.1):
#     """Dehaze an image using the AOD-Net inspired approach."""
#     # Convert image to float32 for calculations
#     image = image.astype(np.float32) / 255.0

#     # Compute dark channel
#     dark_channel = np.min(image, axis=2)

#     # Estimate atmospheric light
#     top_percent = 0.001  # Top 0.1% brightest pixels
#     num_pixels = int(dark_channel.size * top_percent)
#     indices = np.argpartition(dark_channel.ravel(), -num_pixels)[-num_pixels:]
#     atmospheric_light = np.max(image.reshape(-1, 3)[indices], axis=0)

#     # Compute transmission map
#     transmission = 1 - omega * (dark_channel / atmospheric_light.max())
#     transmission = np.clip(transmission, t0, 1.0)

#     # Recover the scene radiance
#     dehazed_image = np.zeros_like(image)
#     for i in range(3):
#         dehazed_image[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]

#     # Clip and convert back to uint8
#     dehazed_image = np.clip(dehazed_image, 0, 1)
#     dehazed_image = (dehazed_image * 255).astype(np.uint8)

#     return dehazed_image

# # Post-dehazing enhancement
# def enhance_image(image):
#     """Enhance the image after dehazing."""
#     # Convert to LAB color space for contrast enhancement
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     # Apply CLAHE to the L channel
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)

#     # Merge the channels and convert back to BGR
#     lab = cv2.merge((l, a, b))
#     enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#     # Sharpen the image
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     sharpened_image = cv2.filter2D(enhanced_image, -1, kernel)

#     return sharpened_image

# # Preprocess image for better detection
# def preprocess_image(image):
#     # Resize image
#     image = cv2.resize(image, (640, 640))

#     # Enhance contrast
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)
#     lab = cv2.merge((l, a, b))
#     image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#     return image

# # Detect number plate region using YOLOv8
# def detect_number_plate_region(image):
#     """Detect the number plate region using YOLOv8."""
#     # Preprocess image
#     processed_image = preprocess_image(image)

#     # Perform inference with lower confidence threshold
#     results = yolov8_model(processed_image, conf=0.3)  # Adjust confidence threshold
#     plates = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
#     if len(plates) > 0:
#         x1, y1, x2, y2 = plates[0]  # Get the first detected plate
#         return int(x1), int(y1), int(x2), int(y2)
#     return None

# # Filter OCR results to extract the most likely number plate text
# def filter_number_plate_text(text):
#     """Filter OCR results to extract the most likely number plate text."""
#     # Use regex to find alphanumeric sequences that resemble number plates
#     matches = re.findall(r'\b[A-Z0-9]{6,10}\b', text)  # Adjust regex as needed
#     if matches:
#         return matches[0]  # Return the first match
#     return "No valid number plate text detected"

# # Number plate detection and OCR
# def detect_number_plate(image_path):
#     image = cv2.imread(image_path)
#     plate_region = detect_number_plate_region(image)
#     if plate_region:
#         x1, y1, x2, y2 = plate_region
#         cropped_plate = image[y1:y2, x1:x2]

#         # Preprocess the cropped plate (optional: resize, sharpen, etc.)
#         cropped_plate = cv2.resize(cropped_plate, (300, 100))  # Resize for better OCR

#         # Perform OCR on the cropped plate
#         result = ocr.ocr(cropped_plate)
#         text = ""
#         if result and len(result) > 0:  # Check if OCR result is not None or empty
#             for line in result:
#                 if line and len(line) > 0:  # Check if line is not None or empty
#                     for word in line:
#                         if word and len(word) > 1:  # Check if word is not None and has text
#                             text += word[1][0] + " "

#         # Filter out unwanted characters and extract the most likely number plate text
#         filtered_text = filter_number_plate_text(text)
#         return filtered_text
#     else:
#         # Fallback: Detect closely packed characters in the entire image
#         result = ocr.ocr(image)
#         text = ""
#         if result and len(result) > 0:
#             for line in result:
#                 if line and len(line) > 0:
#                     for word in line:
#                         if word and len(word) > 1:
#                             text += word[1][0] + " "

#         # Filter out unwanted characters and extract the most likely number plate text
#         filtered_text = filter_number_plate_text(text)
#         return filtered_text

# @app.route('/')
# def index():
#     return "Flask Backend for Number Plate Detection"

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     option = request.form.get('option')

#     if option == '1':
#         # Option 1: Extract number plate
#         number_plate_text = detect_number_plate(file_path)
#         return jsonify({
#             "result": number_plate_text,
#             "image": filename,
#             "dehazed_image": None
#         })

#     elif option == '2':
#         # Option 2: Dehaze image and extract number plate
#         image = cv2.imread(file_path)
#         dehazed_image = aod_dehaze(image)  # Use the aod_dehaze function
#         enhanced_image = enhance_image(dehazed_image)  # Enhance the dehazed image
#         dehazed_path = os.path.join(app.config['RESULT_FOLDER'], f"dehazed_{filename}")
#         cv2.imwrite(dehazed_path, enhanced_image)

#         # Try detection on the enhanced dehazed image
#         number_plate_text = detect_number_plate(dehazed_path)
#         if number_plate_text == "No valid number plate text detected":
#             # Fallback to normal detection if dehazed detection fails
#             number_plate_text = detect_number_plate(file_path)

#         return jsonify({
#             "result": number_plate_text,
#             "image": filename,
#             "dehazed_image": f"dehazed_{filename}"
#         })

#     else:
#         return jsonify({"error": "Invalid option"}), 400

# @app.route('/uploads/<filename>')
# def get_uploaded_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/results/<filename>')
# def get_result_image(filename):
#     return send_from_directory(app.config['RESULT_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)











# from flask import Flask, request, jsonify, send_from_directory
# import cv2
# import numpy as np
# import os
# import re
# import torch
# from paddleocr import PaddleOCR
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# import logging
# from ultralytics import YOLO  # YOLOv8

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # File upload and result directories
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Load YOLOv8 model for license plate detection
# def load_yolov8_model():
#     model = YOLO('yolov8n.pt')  # Replace with custom YOLOv8 model if needed
#     return model

# yolov8_model = load_yolov8_model()

# # Deep Learning-Based Dehazing (Example: Replace with AOD-Net or FFA-Net)
# def dehaze_image(image):
#     """
#     Apply deep learning-based dehazing to the image.
#     Replace this with a pre-trained dehazing model like AOD-Net.
#     """
#     # Placeholder: Return the original image (replace with actual dehazing logic)
#     return image

# # Detect number plate region using YOLOv8
# def detect_number_plate_region(image):
#     """
#     Detect the number plate region using YOLOv8.
#     """
#     results = yolov8_model(image)  # Run inference
#     plates = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
#     if len(plates) > 0:
#         x1, y1, x2, y2 = plates[0]  # Get the first detected plate
#         return int(x1), int(y1), int(x2), int(y2)
#     return None

# # Extract alphanumeric groups from OCR result
# def extract_alphanumeric_groups(text):
#     """
#     Extract alphanumeric groups (e.g., license plate-like patterns) from OCR text.
#     """
#     # Regex to match alphanumeric groups (e.g., "ABC123" or "AB 1234")
#     pattern = r'\b[A-Za-z0-9]+\b'
#     matches = re.findall(pattern, text)
#     return set(matches)  # Return unique groups

# # Number plate detection and OCR
# def detect_number_plate(image_path):
#     """
#     Detect and extract text from the number plate.
#     If no plate is detected, perform OCR on the entire image.
#     """
#     try:
#         image = cv2.imread(image_path)
#         plate_region = detect_number_plate_region(image)
#         if plate_region:
#             # Crop the detected plate region
#             x1, y1, x2, y2 = plate_region
#             cropped_plate = image[y1:y2, x1:x2]
#             result = ocr.ocr(cropped_plate)
#             text = ""
#             for line in result:
#                 for word in line:
#                     text += word[1][0] + " "

#             # Extract alphanumeric groups
#             filtered_text = extract_alphanumeric_groups(text)
#             return filtered_text
#         else:
#             # If no plate is detected, perform OCR on the entire image
#             result = ocr.ocr(image)
#             text = ""
#             for line in result:
#                 for word in line:
#                     text += word[1][0] + " "

#             # Extract alphanumeric groups
#             filtered_text = extract_alphanumeric_groups(text)
#             return filtered_text
#     except Exception as e:
#         logger.error(f"Error in detect_number_plate: {e}")
#         return {"error": "Error processing image"}

# # Flask Routes
# @app.route('/')
# def index():
#     return "Flask Backend for Number Plate Detection"

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     option = request.form.get('option')

#     if option == '1':
#         # Option 1: Extract number plate
#         number_plate_text = detect_number_plate(file_path)
#         return jsonify({
#             "result": list(number_plate_text) if isinstance(number_plate_text, set) else number_plate_text,
#             "image": filename,
#             "dehazed_image": None
#         })

#     elif option == '2':
#         # Option 2: Dehaze image and extract number plate
#         try:
#             image = cv2.imread(file_path)
#             dehazed_image = dehaze_image(image)
#             dehazed_path = os.path.join(app.config['RESULT_FOLDER'], f"dehazed_{filename}")
#             cv2.imwrite(dehazed_path, dehazed_image)

#             number_plate_text = detect_number_plate(dehazed_path)
#             return jsonify({
#                 "result": list(number_plate_text) if isinstance(number_plate_text, set) else number_plate_text,
#                 "image": filename,
#                 "dehazed_image": f"dehazed_{filename}"
#             })
#         except Exception as e:
#             logger.error(f"Error in dehazing: {e}")
#             return jsonify({"error": "Error processing image"}), 500

#     else:
#         return jsonify({"error": "Invalid option"}), 400

# @app.route('/uploads/<filename>')
# def get_uploaded_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/results/<filename>')
# def get_result_image(filename):
#     return send_from_directory(app.config['RESULT_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)














# from flask import Flask, request, jsonify, send_from_directory
# import cv2
# import numpy as np
# import os
# import re
# import torch
# from paddleocr import PaddleOCR
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# import logging
# from ultralytics import YOLO  # YOLOv8

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # File upload and result directories
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Load YOLOv8 model for license plate detection
# def load_yolov8_model():
#     model = YOLO('yolov8n.pt')  # Replace with custom YOLOv8 model if needed
#     return model

# yolov8_model = load_yolov8_model()

# # Load AOD-Net Dehazing Model (Placeholder)
# def load_aod_net_model():
#     """
#     Load a pre-trained AOD-Net model for dehazing.
#     Replace this with your actual AOD-Net implementation.
#     """
#     model_path = 'aod_net.pth'  # Replace with the path to your AOD-Net model
#     if not os.path.exists(model_path):
#         logger.warning(f"AOD-Net model not found at {model_path}. Using placeholder.")
#         return None
    
#     # Load the model (example for PyTorch)
#     try:
#         model = torch.load(model_path, map_location=torch.device('cpu'))
#         model.eval()  # Set the model to evaluation mode
#         logger.info("AOD-Net model loaded successfully.")
#         return model
#     except Exception as e:
#         logger.error(f"Error loading AOD-Net model: {e}")
#         return None

# aod_net_model = load_aod_net_model()

# # Deep Learning-Based Dehazing using AOD-Net
# def dehaze_image(image):
#     """
#     Apply deep learning-based dehazing to the image using AOD-Net.
#     If AOD-Net is not available, return the original image.
#     """
#     if aod_net_model is None:
#         logger.warning("AOD-Net model not loaded. Returning original image.")
#         return image
    
#     try:
#         # Preprocess the image for AOD-Net
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#         image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
#         image = np.transpose(image, (2, 0, 1))  # Change to CxHxW format
#         image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension

#         # Perform dehazing using AOD-Net
#         with torch.no_grad():
#             dehazed_image = aod_net_model(image)

#         # Postprocess the dehazed image
#         dehazed_image = dehazed_image.squeeze(0).cpu().numpy()  # Remove batch dimension
#         dehazed_image = np.transpose(dehazed_image, (1, 2, 0))  # Change back to HxWxC format
#         dehazed_image = (dehazed_image * 255).astype(np.uint8)  # Scale back to [0, 255]
#         dehazed_image = cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR

#         return dehazed_image
#     except Exception as e:
#         logger.error(f"Error in dehazing: {e}")
#         return image  # Return the original image if dehazing fails

# # Detect number plate region using YOLOv8
# def detect_number_plate_region(image):
#     """
#     Detect the number plate region using YOLOv8.
#     """
#     results = yolov8_model(image)  # Run inference
#     plates = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
#     if len(plates) > 0:
#         x1, y1, x2, y2 = plates[0]  # Get the first detected plate
#         return int(x1), int(y1), int(x2), int(y2)
#     return None

# # Extract alphanumeric groups from OCR result
# def extract_alphanumeric_groups(text):
#     """
#     Extract alphanumeric groups (e.g., license plate-like patterns) from OCR text.
#     """
#     # Regex to match alphanumeric groups (e.g., "ABC123" or "AB 1234")
#     pattern = r'\b[A-Za-z0-9]+\b'
#     matches = re.findall(pattern, text)
#     return set(matches)  # Return unique groups

# # Number plate detection and OCR
# def detect_number_plate(image_path):
#     """
#     Detect and extract text from the number plate.
#     If no plate is detected, perform OCR on the entire image.
#     """
#     try:
#         image = cv2.imread(image_path)
#         plate_region = detect_number_plate_region(image)
#         if plate_region:
#             # Crop the detected plate region
#             x1, y1, x2, y2 = plate_region
#             cropped_plate = image[y1:y2, x1:x2]
#             result = ocr.ocr(cropped_plate)
#             text = ""
#             for line in result:
#                 for word in line:
#                     text += word[1][0] + " "

#             # Extract alphanumeric groups
#             filtered_text = extract_alphanumeric_groups(text)
#             return filtered_text
#         else:
#             # If no plate is detected, perform OCR on the entire image
#             result = ocr.ocr(image)
#             text = ""
#             for line in result:
#                 for word in line:
#                     text += word[1][0] + " "

#             # Extract alphanumeric groups
#             filtered_text = extract_alphanumeric_groups(text)
#             return filtered_text
#     except Exception as e:
#         logger.error(f"Error in detect_number_plate: {e}")
#         return {"error": "Error processing image"}

# # Flask Routes
# @app.route('/')
# def index():
#     return "Flask Backend for Number Plate Detection"

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     option = request.form.get('option')

#     if option == '1':
#         # Option 1: Extract number plate
#         number_plate_text = detect_number_plate(file_path)
#         return jsonify({
#             "result": list(number_plate_text) if isinstance(number_plate_text, set) else number_plate_text,
#             "image": filename,
#             "dehazed_image": None
#         })

#     elif option == '2':
#         # Option 2: Dehaze image and extract number plate
#         try:
#             image = cv2.imread(file_path)
#             dehazed_image = dehaze_image(image)
#             dehazed_path = os.path.join(app.config['RESULT_FOLDER'], f"dehazed_{filename}")
#             cv2.imwrite(dehazed_path, dehazed_image)

#             number_plate_text = detect_number_plate(dehazed_path)
#             return jsonify({
#                 "result": list(number_plate_text) if isinstance(number_plate_text, set) else number_plate_text,
#                 "image": filename,
#                 "dehazed_image": f"dehazed_{filename}"
#             })
#         except Exception as e:
#             logger.error(f"Error in dehazing: {e}")
#             return jsonify({"error": "Error processing image"}), 500

#     else:
#         return jsonify({"error": "Invalid option"}), 400

# @app.route('/uploads/<filename>')
# def get_uploaded_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/results/<filename>')
# def get_result_image(filename):
#     return send_from_directory(app.config['RESULT_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)