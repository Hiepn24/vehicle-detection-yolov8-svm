from flask import Flask, request, render_template, redirect, url_for, Response
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from ultralytics import YOLO
import base64

# --- Khởi tạo Flask App ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Tải model SVM và scaler ---
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- Tải model YOLO ---
yolo_model = YOLO("model.pt")

# --- Biến toàn cục cho camera ---
camera_on = False
cap = None

# --- Xử lý ảnh ---
def load_and_preprocess_image(image_path, size=(96, 96)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    return img

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    return features

# --- Trang chủ ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Nhận diện phương tiện bằng SVM ---
@app.route('/detect-vehicle', methods=['GET', 'POST'])
def detect_vehicle():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return "Bạn chưa chọn file hợp lệ!"

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        img = load_and_preprocess_image(image_path)
        features = extract_features(img).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

    return render_template('detect_vehicle.html', prediction=prediction, image_path=image_path)

# --- Nhận diện ảnh tĩnh bằng YOLO ---
@app.route('/detect-yolo', methods=['GET', 'POST'])
def detect_yolo():
    global camera_on
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return "Bạn chưa chọn file!"
        
        in_memory_file = file.read()
        npimg = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Dự đoán bằng YOLO
        results = yolo_model(img)
        img_with_boxes = results[0].plot() # .plot() để vẽ bounding boxes lên ảnh gốc.

        # Encode ảnh thành base64 để truyền về frontend
        _, buffer = cv2.imencode('.jpg', img_with_boxes)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template('detect_yolo.html', image_data=img_base64, camera_on=camera_on)
    
    return render_template('detect_yolo.html', camera_on=camera_on)

# --- Nhận diện bằng video/camera (YOLO + Flask stream) ---
@app.route('/detect-yolo-video')
def detect_yolo_video():
    def generate_frames():
        global cap, camera_on

        cap = cv2.VideoCapture(0)
        frame_count = 0
        prev_frame_with_bbox = None

        while camera_on and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            small_frame = cv2.resize(frame, (416, 416))

            # Chỉ update bounding box mỗi vài frame
            if frame_count % 3 == 0:
                results = yolo_model(small_frame)
                detections = results[0].plot()
                detections = cv2.resize(detections, (frame.shape[1], frame.shape[0]))
                prev_frame_with_bbox = detections  # Cập nhật frame mới

            # Nếu có frame đã detect thì hiển thị
            if prev_frame_with_bbox is not None:
                display_frame = prev_frame_with_bbox
            else:
                display_frame = frame

            _, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if cap:
            cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-camera')
def start_camera():
    global camera_on
    camera_on = True
    return redirect(url_for('detect_yolo'))

@app.route('/stop-camera')
def stop_camera():
    global camera_on
    camera_on = False
    if cap:
        cap.release()
    return redirect(url_for('detect_yolo'))

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    app.run(debug=True)
