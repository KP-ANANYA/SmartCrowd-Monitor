from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, send_from_directory
import json
import cv2
import numpy as np  # ensure numpy is imported
from ultralytics import YOLO
from datetime import datetime
import winsound
import threading
import time
import os
import hashlib
import sqlite3
from functools import wraps
from firebase_config import firebase_config
from collections import deque
from werkzeug.utils import secure_filename
import math
import uuid
import queue
import zone_polygon_utils as zutil 
import smtplib
from email.mime.text import MIMEText

try:
    import torch
except Exception as _torch_err:
    torch = None
    print(f"⚠️ Torch not available yet ({_torch_err}); YOLO will run on CPU. Install torch for better performance.")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DOWNLOADS_DIR = os.path.join(BASE_DIR, 'downloads')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
zone_config = zutil.load_zone_polygons(BASE_DIR)
heatmap_acc = None

# Store currently logged-in user's email
CURRENT_USER_EMAIL = None

# Limit uploads to ~500MB to avoid abuse
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Configure device for YOLO and threading
CUDA_AVAILABLE = False
DEVICE = 'cpu'
USE_HALF = False
if torch is not None:
    try:
        CUDA_AVAILABLE = torch.cuda.is_available()
        DEVICE = 0 if CUDA_AVAILABLE else 'cpu'
        USE_HALF = True if CUDA_AVAILABLE else False
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    except Exception as _cuda_err:
        print(f"⚠️ Torch threading/CUDA init issue: {_cuda_err}")

# Initialize database (fallback to SQLite if Firebase fails)
def init_db():
    try:
        # Test Firebase connection
        if firebase_config.db is not None:
            print("✅ Using Firebase for user storage")
            return
    except Exception as e:
        print(f"⚠️ Firebase not available, using SQLite fallback: {e}")
    
    # Fallback to SQLite
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()
    print("✅ Using SQLite for user storage")

# Load configuration
def load_config():
    try:
        # Prefer backend/config.json; fallback to root/config.json
        cfg_path = os.path.join(BASE_DIR, 'config.json')
        if not os.path.exists(cfg_path):
            alt = os.path.join(ROOT_DIR, 'config.json')
            cfg_path = alt if os.path.exists(alt) else cfg_path
        with open(cfg_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "camera_settings": {"camera_index": 1, "width": 1920, "height": 1080},
            "detection_settings": {"model_path": "yolov8s.pt", "grid_size": {"rows": 3, "cols": 3}},
            "zone_thresholds": {"low": 0, "medium": 1, "high": 2},
            "alert_settings": {"enable_sound": True, "log_file": "alerts_log.txt"}
        }

config = load_config()

# Global variables for crowd detection
# Resolve YOLO model path relative to backend/ first, then project root
_model_path = config["detection_settings"]["model_path"]
_try_model_paths = [
    os.path.join(BASE_DIR, _model_path),
    os.path.join(ROOT_DIR, _model_path),
    _model_path,
]
_model_path = next((p for p in _try_model_paths if os.path.exists(p)), _model_path)
model = YOLO(_model_path)
try:
    if CUDA_AVAILABLE:
        model.to('cuda')
        # Fuse model layers for small speed boost
        if hasattr(model, 'fuse'):
            model.fuse()
except Exception:
    pass
cap = None
current_zone_data = {"total": 0, "zones": []}
alerted_zones = set()
is_streaming = False
camera_active = False
# New: in-memory alerts/state and lock
alerts_log = deque(maxlen=200)
last_alert_state = {}
state_lock = threading.Lock()

# New: capture/stream resilience state
_capture_lock = threading.Lock()
_read_fail_count = 0
_READ_FAIL_REINIT_THRESHOLD = 15
_last_frame_jpeg = None
_placeholder_jpeg = None

# Upload video live sessions
upload_sessions = {}
upload_lock = threading.Lock()

# Single active upload analysis (HTML provided expects a single ongoing analysis)
upload_analysis_active = False
upload_analysis_stop = False
upload_analysis_thread = None
upload_analysis_path = None

# Inference input size for speed (smaller is faster)
FRAME_SKIP = 2  # Skip every other frame for faster apparent motion (detection every 2nd frame)
FRAME_WIDTH = 480   # legacy small inference size (kept for warmup / fallback)
FRAME_HEIGHT = 270

# Confidence threshold (fall back to 0.35 if not configured). Clamp to a sane range.
CONFIDENCE_THRESHOLD = config.get("detection_settings", {}).get("confidence_threshold", 0.35)
try:
    CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD)
except Exception:
    CONFIDENCE_THRESHOLD = 0.35
CONFIDENCE_THRESHOLD = max(0.05, min(CONFIDENCE_THRESHOLD, 0.9))

UPLOAD_CONFIDENCE_THRESHOLD = float(config.get("detection_settings", {}).get("upload_confidence_threshold", 0.25))
UPLOAD_CONFIDENCE_THRESHOLD = max(0.05, min(UPLOAD_CONFIDENCE_THRESHOLD, 0.9))
UPLOAD_INFERENCE_IMG_SIZE = int(config.get("detection_settings", {}).get("upload_imgsz", 768))
if UPLOAD_INFERENCE_IMG_SIZE < 256 or UPLOAD_INFERENCE_IMG_SIZE > 1280:
    UPLOAD_INFERENCE_IMG_SIZE = 768
UPLOAD_TILE_DETECTION = bool(config.get("detection_settings", {}).get("upload_tile_detection", True))
UPLOAD_TILE_OVERLAP = float(config.get("detection_settings", {}).get("upload_tile_overlap", 0.20))
UPLOAD_MIN_BOX_AREA = int(config.get("detection_settings", {}).get("upload_min_box_area", 80))

# Inference image size (smaller -> faster). Allow override via config detection_settings.imgsz
INFERENCE_IMG_SIZE = int(config.get("detection_settings", {}).get("imgsz", 512))
if INFERENCE_IMG_SIZE < 256 or INFERENCE_IMG_SIZE > 1280:
    INFERENCE_IMG_SIZE = 512

# High accuracy toggle (inline inference bypassing async queue)
HIGH_ACCURACY = config.get("detection_settings", {}).get("high_accuracy", True)
SIMPLE_MODE = config.get("detection_settings", {}).get("simple_mode", True)  # if True, mimic provided reference script exactly
DEBUG_DETECTION = config.get("detection_settings", {}).get("debug_detection", True)
PURE_SIMPLE = config.get("detection_settings", {}).get("pure_simple", False)  # strongest simplification: direct model(frame) only
ADAPTIVE_ENABLED = True  # auto-adjust confidence if repeated zero detections

# Additional filtering parameters to reduce false positives / double counts
# Tuned for fewer duplicate boxes while avoiding merging distinct nearby people.
MIN_BOX_AREA = 300           # ignore very tiny boxes (noise / partial limbs)
IOU_DEDUP_THRESHOLD = 0.8    # only merge if boxes overlap strongly
CENTER_DIST_DEDUP = 25        # px distance under which centers are considered same person

# Live adaptive state
_zero_streak = 0  # counts consecutive frames with zero detections (simple mode)
latest_boxes = []  # list of (x1,y1,x2,y2,conf) for last processed frame
adaptive_conf = CONFIDENCE_THRESHOLD  # live adjustable confidence
zero_frame_streak = 0

# Warm up YOLO model once to reduce first-inference delay
_model_warmed = False

def _warmup_model():
    global _model_warmed
    if _model_warmed:
        return
    try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model.predict(dummy, imgsz=INFERENCE_IMG_SIZE, classes=[0], conf=0.2, verbose=False)
            _model_warmed = True
            print("⚡ YOLO model warmed up")
    except Exception as e:
        print(f"⚠️ YOLO warmup skipped: {e}")

# Efficient live detection: threaded YOLO inference, frame skipping, resize
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def _compute_iou(a, b):
    # a,b: (x1,y1,x2,y2)
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    area_b = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _deduplicate_boxes(boxes, min_area=MIN_BOX_AREA):
    """Given list of (x1,y1,x2,y2,conf) return filtered unique boxes."""
    # Remove tiny boxes
    filtered = []
    for b in boxes:
        w = max(0, b[2]-b[0]); h = max(0, b[3]-b[1])
        if w * h < min_area:
            continue
        filtered.append(b)
    # Sort by confidence desc
    filtered.sort(key=lambda x: x[4], reverse=True)
    kept = []
    for b in filtered:
        bx_cx = (b[0]+b[2]) / 2.0; bx_cy = (b[1]+b[3]) / 2.0
        duplicate = False
        for k in kept:
            iou = _compute_iou(b, k)
            if iou >= IOU_DEDUP_THRESHOLD:
                duplicate = True; break
            kc_x = (k[0]+k[2]) / 2.0; kc_y = (k[1]+k[3]) / 2.0
            if abs(kc_x - bx_cx) < CENTER_DIST_DEDUP and abs(kc_y - bx_cy) < CENTER_DIST_DEDUP:
                duplicate = True; break
        if not duplicate:
            kept.append(b)
    return kept

def _extract_person_boxes(results, offset_x=0, offset_y=0):
    boxes = []
    for r in results:
        for box in getattr(r, 'boxes', []) or []:
            try:
                cls = int(box.cls[0])
            except Exception:
                cls = -1
            if cls != 0:
                continue
            try:
                conf = float(box.conf[0]) if getattr(box, 'conf', None) is not None else 0.0
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, conf))
            except Exception:
                pass
    return boxes

def _detect_people_upload(frame):
    raw_boxes = []
    try:
        results = model.predict(
            frame,
            imgsz=UPLOAD_INFERENCE_IMG_SIZE,
            conf=UPLOAD_CONFIDENCE_THRESHOLD,
            classes=[0],
            device=DEVICE,
            half=USE_HALF,
            verbose=False
        )
        raw_boxes.extend(_extract_person_boxes(results))
    except Exception as e:
        return [], f'inference error: {e}'

    if UPLOAD_TILE_DETECTION:
        h, w = frame.shape[:2]
        tile_w = max(320, int(w * 0.55))
        tile_h = max(320, int(h * 0.55))
        step_x = max(1, int(tile_w * (1.0 - UPLOAD_TILE_OVERLAP)))
        step_y = max(1, int(tile_h * (1.0 - UPLOAD_TILE_OVERLAP)))
        x_positions = list(range(0, max(1, w - tile_w + 1), step_x))
        y_positions = list(range(0, max(1, h - tile_h + 1), step_y))
        if not x_positions or x_positions[-1] != max(0, w - tile_w):
            x_positions.append(max(0, w - tile_w))
        if not y_positions or y_positions[-1] != max(0, h - tile_h):
            y_positions.append(max(0, h - tile_h))

        for y0 in y_positions:
            for x0 in x_positions:
                crop = frame[y0:min(y0 + tile_h, h), x0:min(x0 + tile_w, w)]
                if crop.size == 0:
                    continue
                try:
                    tile_results = model.predict(
                        crop,
                        imgsz=UPLOAD_INFERENCE_IMG_SIZE,
                        conf=UPLOAD_CONFIDENCE_THRESHOLD,
                        classes=[0],
                        device=DEVICE,
                        half=USE_HALF,
                        verbose=False
                    )
                    raw_boxes.extend(_extract_person_boxes(tile_results, x0, y0))
                except Exception:
                    pass

    return _deduplicate_boxes(raw_boxes, min_area=UPLOAD_MIN_BOX_AREA), None

def yolo_worker():
    """Background thread: pulls latest frame and runs YOLO with better accuracy (no forced distortion)."""
    while camera_active:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        original = frame
        h, w = original.shape[:2]
        scale_factor = 1.0
        proc = original
        max_dim = max(w, h)
        try:
            if max_dim > 960:  # limit very large frames for perf
                scale_factor = 960.0 / max_dim
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                proc = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            proc = original
            scale_factor = 1.0

        try:
            results = model.predict(
                proc,
                imgsz=INFERENCE_IMG_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                classes=[0],
                device=DEVICE,
                half=USE_HALF,
                verbose=False
            )
        except Exception:
            results = []

        # Adaptive secondary pass if zero people and threshold > 0.28 (avoid missing distant people)
        try:
            people_found = 0
            for r in results:
                people_found += int((r.boxes.cls == 0).sum().item()) if getattr(r, 'boxes', None) is not None else 0
            if people_found == 0 and CONFIDENCE_THRESHOLD > 0.28:
                try:
                    results = model.predict(
                        proc,
                        imgsz=INFERENCE_IMG_SIZE,
                        conf=0.28,
                        classes=[0],
                        device=DEVICE,
                        half=USE_HALF,
                        verbose=False
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # Keep only latest result
        try:
            while not result_queue.empty():
                result_queue.get_nowait()
        except Exception:
            pass
        result_queue.put((original, results, scale_factor))

def _ensure_placeholder_frame(width=640, height=480):
    global _placeholder_jpeg
    if _placeholder_jpeg is not None:
        return _placeholder_jpeg
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, 'No camera frame available', (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode('.jpg', img)
    if ok:
        _placeholder_jpeg = buf.tobytes()
    return _placeholder_jpeg


# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authentication routes
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        
        # Try Firebase first
        if firebase_config.db is not None:
            try:
                users_ref = firebase_config.db.collection('users')
                query = users_ref.where('username', '==', username).limit(1)
                users = query.get()
                
                if users:
                    user_doc = users[0]
                    user_data = user_doc.to_dict()
                    if user_data.get('password_hash') == hash_password(password):
                        global CURRENT_USER_EMAIL

                        session['user_id'] = user_doc.id
                        session['username'] = user_data['username']
                        session['email'] = user_data['email']

                        CURRENT_USER_EMAIL = user_data['email']

                        flash('Login successful! Welcome to SmartCrowd Monitor.', 'success')
                        return redirect(url_for('index'))
                    else:
                        flash('Invalid username or password!', 'error')
                else:
                    flash('Invalid username or password!', 'error')
            except Exception as e:
                print(f"Firebase login error: {e}")
                flash('Login service temporarily unavailable. Please try again.', 'error')
        else:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT id, username, password FROM users WHERE username = ?", (username,))
            user = c.fetchone()
            conn.close()
            
            if user and user[2] == hash_password(password):
                session['user_id'] = user[0]
                session['username'] = user[1]
                flash('Login successful! Welcome to SmartCrowd Monitor.', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        
        if firebase_config.db is not None:
            try:
                user_data = {
                    'username': username,
                    'email': email,
                    'password_hash': hash_password(password),
                    'created_at': datetime.now(),
                    'is_active': True
                }
                users_ref = firebase_config.db.collection('users')
                if users_ref.where('username', '==', username).limit(1).get():
                    flash('Username already exists!', 'error')
                    return render_template('register.html')
                if users_ref.where('email', '==', email).limit(1).get():
                    flash('Email already exists!', 'error')
                    return render_template('register.html')
                firebase_config.db.collection('users').add(user_data)
                flash('Registration successful! Please login to start monitoring.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                print(f"Firebase registration error: {e}")
                flash('Registration service temporarily unavailable. Please try again.', 'error')
                return render_template('register.html')
        else:
            try:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                         (username, email, hash_password(password)))
                conn.commit()
                conn.close()
                flash('Registration successful! Please login to start monitoring.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username or email already exists!', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('dashboard'))

@app.route('/monitoring')
@login_required
def index():
    return render_template('monitoring.html')

@app.route('/index')
@login_required
def monitoring():
    return render_template('monitoring.html')
@app.route('/zone-editor')
@login_required
def zone_editor_page():
    return render_template('zone_editor.html')


@app.route('/zone_snapshot')
@login_required
def zone_snapshot():
    """One still frame for the editor canvas to draw on top of."""
    global cap
    with _capture_lock:
        if cap is None or not cap.isOpened():
            ok, frame = False, None
        else:
            ok, frame = cap.read()
    if not ok or frame is None:
        return Response(_ensure_placeholder_frame(), mimetype='image/jpeg')
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/save_zones', methods=['POST'])
@login_required
def save_zones_api():
    global zone_config
    data = request.get_json(force=True)
    zones = data.get('zones', [])
    fw, fh = data.get('frame_width'), data.get('frame_height')
    if not zones or not fw or not fh:
        return jsonify({"error": "Missing zones or frame size"}), 400
    zone_config = zutil.save_zone_polygons(BASE_DIR, zones, fw, fh)
    return jsonify({"status": "ok", "zones_saved": len(zones)})
# Crowd detection functions

# Initialize zeroed zone data using configured grid
def _init_zero_zone_data():
    grid_size = config["detection_settings"]["grid_size"]
    rows, cols = grid_size["rows"], grid_size["cols"]
    zones = []
    for r in range(rows):
        for c in range(cols):
            zones.append({"id": f"Z{r * cols + c + 1}", "count": 0, "level": "Low"})
    return {"total": 0, "zones": zones}

def initialize_camera():
    global cap, is_streaming, camera_active, _read_fail_count, _last_frame_jpeg, current_zone_data
    cam_settings = config["camera_settings"]

    tried = set()
    try_indices = [cam_settings.get("camera_index", 0), 0, 1, 2]
    opened = False
    for idx in try_indices:
        if idx in tried:
            continue
        tried.add(idx)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        # Reduce camera latency and resolution for faster counting
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if SIMPLE_MODE:
            # Use configured resolution directly (like reference script)
            target_w = cam_settings.get("width", 1920)
            target_h = cam_settings.get("height", 1080)
        else:
            target_w = min(cam_settings.get("width", 1280), 640)
            target_h = min(cam_settings.get("height", 720), 360)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

        warm_ok = False
        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                warm_ok = True
                break
            time.sleep(0.05)
        if cap.isOpened() and warm_ok:
            print(f"✅ Camera initialized on index {idx}")
            opened = True
            break
        else:
            try:
                cap.release()
            except Exception:
                pass
            cap = None

    if not opened:
        print("❌ Error: Could not open webcam on any tried index.")
        camera_active = False
        return False

    camera_active = True
    is_streaming = True
    _read_fail_count = 0
    _last_frame_jpeg = None
    # initialize default grid state so UI shows zones immediately
    with state_lock:
        current_zone_data = _init_zero_zone_data()
    # Warm up inference now to avoid first frame delay
    _warmup_model()
    print("✅ Camera initialized and monitoring started")
    return True


def _maybe_reinit_camera():
    """Try to reinitialize camera if it got closed while streaming."""
    global cap
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    time.sleep(0.2)
    return initialize_camera()


def stop_camera():
    global cap, is_streaming, camera_active
    is_streaming = False
    camera_active = False
    if cap is not None:
        try:
            cap.release()
        finally:
            cap = None
    print("🛑 Camera stopped and monitoring ended")
    
def send_alert_email(zone_id, zone_count, level):

    global CURRENT_USER_EMAIL

    alert_settings = config.get("alert_settings", {})

    # Check if email alerts are enabled
    if not alert_settings.get("enable_email", False):
        return

    receiver_email = CURRENT_USER_EMAIL

    if receiver_email is None:
        print("❌ No logged-in user email found.")
        return

    sender_email = alert_settings["sender_email"]
    sender_password = alert_settings["sender_app_password"]

    subject = "🚨 SmartCrowd Monitor Alert"

    body = f"""
Crowd Threshold Exceeded!

Zone ID      : {zone_id}
Crowd Level  : {level}
People Count : {zone_count}

Time:
{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

Please check SmartCrowd Monitor immediately.
"""

    msg = MIMEText(body)

    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP(
            alert_settings["smtp_server"],
            alert_settings["smtp_port"]
        )

        server.starttls()

        server.login(
            sender_email,
            sender_password
        )

        server.sendmail(
            sender_email,
            receiver_email,
            msg.as_string()
        )

        server.quit()

        print(f"📧 Alert email sent to {receiver_email}")

    except Exception as e:
        print("❌ Email Error:", e)


def process_frame():
    global current_zone_data, alerted_zones, last_alert_state, _read_fail_count, adaptive_conf, zero_frame_streak
    if not camera_active:
        return None, None
    with _capture_lock:
        if cap is None or not cap.isOpened():
            _read_fail_count += 1
            if _read_fail_count >= _READ_FAIL_REINIT_THRESHOLD and camera_active:
                print("♻️ Attempting to reinitialize camera after repeated failures...")
                _read_fail_count = 0
                _maybe_reinit_camera()
            return None, None
        success, frame = cap.read()
    if not success or frame is None:
        _read_fail_count += 1
        if _read_fail_count >= _READ_FAIL_REINIT_THRESHOLD and camera_active:
            print("♻️ Attempting to reinitialize camera after repeated read failures...")
            _read_fail_count = 0
            _maybe_reinit_camera()
        return None, None
    _read_fail_count = 0

    # Frame skipping for speed
    if hasattr(process_frame, 'frame_count'):
        process_frame.frame_count += 1
    else:
        process_frame.frame_count = 0
    if process_frame.frame_count % FRAME_SKIP != 0:
        return frame, current_zone_data

    if HIGH_ACCURACY:
        # Inline inference for freshest frame (less lag, better spatial alignment)
        if SIMPLE_MODE:
            # Explicit predict call so we control conf & imgsz for consistency; single pass only for stability
            try:
                results = model.predict(
                    frame,
                    imgsz=INFERENCE_IMG_SIZE,
                    conf=adaptive_conf,
                    classes=[0],
                    device=DEVICE,
                    half=USE_HALF,
                    verbose=False
                )
            except Exception as e:
                print(f"⚠️ Inline simple inference error: {e}")
                results = []
        else:
            try:
                results = model.predict(
                    frame,
                    imgsz=INFERENCE_IMG_SIZE,
                    conf=adaptive_conf,
                    classes=[0],
                    device=DEVICE,
                    half=USE_HALF,
                    verbose=False
                )
            except Exception as e:
                print(f"⚠️ Inline inference error (conf={CONFIDENCE_THRESHOLD}): {e}")
                results = []
            # Adaptive fallback: if no boxes detected at current threshold, retry with lower threshold
            try:
                no_boxes = True
                for r in results:
                    if getattr(r, 'boxes', None) is not None and len(r.boxes) > 0:
                        no_boxes = False
                        break
                if no_boxes and adaptive_conf > 0.3:
                    low_conf = max(0.25, adaptive_conf - 0.15)
                    try:
                        alt_results = model.predict(
                            frame,
                            imgsz=INFERENCE_IMG_SIZE,
                            conf=low_conf,
                            classes=[0],
                            device=DEVICE,
                            half=USE_HALF,
                            verbose=False
                        )
                        # Use alt_results only if it actually found people
                        found = False
                        for ar in alt_results:
                            if getattr(ar, 'boxes', None) is not None and len(ar.boxes) > 0:
                                found = True
                                break
                        if found:
                            results = alt_results
                            if adaptive_conf - low_conf > 0.05:
                                print(f"ℹ️ Adaptive per-frame fallback used (from {adaptive_conf} to {low_conf})")
                    except Exception as e2:
                        print(f"⚠️ Fallback inference error: {e2}")
            except Exception:
                pass
        frame_out = frame
        scale_factor = None
    else:
        # Threaded YOLO inference (keep only the latest frame in queue, and do not block waiting for results)
        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()
        except Exception:
            pass
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        try:
            queue_item = result_queue.get_nowait()
            if len(queue_item) == 3:
                frame_out, results, scale_factor = queue_item
            else:  # backward compatibility
                frame_out, results = queue_item
                scale_factor = None
        except queue.Empty:
            return frame, current_zone_data
    # --- Existing zone logic, but use frame_out ---
    grid_size = config["detection_settings"]["grid_size"]
    rows, cols = grid_size["rows"], grid_size["cols"]
    height, width = frame_out.shape[:2]
    zone_h, zone_w = height // rows, width // cols
    zone_counts = np.zeros((rows, cols), dtype=int)
    total_count = 0

    # NEW: use drawn polygons if saved, otherwise fall back to the grid
    runtime_zones = zutil.get_runtime_zones(zone_config, width, height) if zone_config else None
    polygon_counts = {z["id"]: 0 for z in runtime_zones} if runtime_zones else None
    centroids = []

    if SIMPLE_MODE:
        total_count = 0
        boxes_collected = []
        for r in results:
            for box in getattr(r, 'boxes', []) or []:
                try:
                    cls = int(box.cls[0])
                except Exception:
                    cls = -1
                if cls != 0:
                    continue
                try:
                    x1f, y1f, x2f, y2f = map(float, box.xyxy[0])
                    if scale_factor and scale_factor != 1.0:
                        inv = 1.0 / scale_factor
                        x1f, y1f, x2f, y2f = x1f * inv, y1f * inv, x2f * inv, y2f * inv
                    x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
                    conf_val = float(box.conf[0]) if getattr(box, 'conf', None) is not None else 0.0
                    total_count += 1
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    centroids.append((cx, cy))
                    if polygon_counts is not None:
                        zid = zutil.assign_point_to_zone(cx, cy, runtime_zones)
                        if zid:
                            polygon_counts[zid] += 1
                    else:
                        row = min(max(cy // zone_h, 0), rows - 1)
                        col = min(max(cx // zone_w, 0), cols - 1)
                        zone_counts[row][col] += 1
                    boxes_collected.append((x1, y1, x2, y2, conf_val))
                except Exception:
                    pass
        global latest_boxes
        latest_boxes = boxes_collected
    else:
        raw_boxes = []  # collect for dedup
        for r in results:
            for box in getattr(r, 'boxes', []) or []:
                try:
                    cls = int(box.cls[0])
                except Exception:
                    cls = -1
                if cls != 0:
                    continue
                try:
                    conf = float(box.conf[0]) if getattr(box, 'conf', None) is not None else 0.0
                    x1f, y1f, x2f, y2f = map(float, box.xyxy[0])
                    if scale_factor and scale_factor != 1.0:
                        inv = 1.0 / scale_factor
                        x1f, y1f, x2f, y2f = x1f * inv, y1f * inv, x2f * inv, y2f * inv
                    raw_boxes.append((max(0,int(x1f)), max(0,int(y1f)), max(0,int(x2f)), max(0,int(y2f)), conf))
                except Exception:
                    continue
        unique_boxes = _deduplicate_boxes(raw_boxes)
        if not unique_boxes and raw_boxes:
            unique_boxes = raw_boxes
        total_count = len(unique_boxes)
        for (x1, y1, x2, y2, conf) in unique_boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centroids.append((cx, cy))
            if polygon_counts is not None:
                zid = zutil.assign_point_to_zone(cx, cy, runtime_zones)
                if zid:
                    polygon_counts[zid] += 1
            else:
                row = min(max(cy // zone_h, 0), rows - 1)
                col = min(max(cx // zone_w, 0), cols - 1)
                zone_counts[row][col] += 1
        latest_boxes = unique_boxes
    frame_data = {"total": int(total_count), "zones": []}

    # Adaptive confidence controller (post-detection)
    if ADAPTIVE_ENABLED:
        if total_count == 0:
            zero_frame_streak += 1
            # Lower confidence progressively after streak thresholds
            if zero_frame_streak in (5, 10, 20):
                new_conf = max(0.15, adaptive_conf - 0.1)
                if new_conf < adaptive_conf:
                    adaptive_conf = round(new_conf, 3)
                    print(f"🔧 Adaptive: lowered confidence to {adaptive_conf} after {zero_frame_streak} zero frames")
        else:
            if zero_frame_streak >= 5 and adaptive_conf < CONFIDENCE_THRESHOLD:
                # Gradually restore toward original threshold
                adaptive_conf = round(min(CONFIDENCE_THRESHOLD, adaptive_conf + 0.05), 3)
                print(f"🔧 Adaptive: restored confidence to {adaptive_conf} (detections present)")
            zero_frame_streak = 0

    if polygon_counts is not None:
        zone_id_count_pairs = list(polygon_counts.items())
    else:
        zone_id_count_pairs = [
            (f"Z{row * cols + col + 1}", int(zone_counts[row][col]))
            for row in range(rows) for col in range(cols)
        ]

    for zone_id, zone_count in zone_id_count_pairs:
        print(f"Zone={zone_id}, Count={zone_count}")
        thresholds = config["zone_thresholds"]

        if zone_count <= thresholds["low"]:
            level = "Low"
        elif zone_count <= thresholds["medium"]:
            level = "Medium"
        elif zone_count <= thresholds["high"]:
            level = "High"
        else:
            level = "Critical"
        print(f"Level={level}")

        frame_data["zones"].append({
            "id": zone_id,
            "count": zone_count,
            "level": level
        })

    # Send alert only when crowd reaches Critical
        if level == "Critical":
            if zone_id not in alerted_zones:
                alerted_zones.add(zone_id)

                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                alert_message = f"[{ts}] ALERT: {zone_id} is in CRITICAL state with {zone_count} people"

                print(alert_message)
                with state_lock:
                    alerts_log.append({
                        "timestamp": ts,
                        "zone": zone_id,
                        "level": "CRITICAL",
                        "count": zone_count,
                        "message": alert_message
                    })

                last_alert_state[zone_id] = {
                    "level": "Critical",
                    "count": zone_count
                }

                send_alert_email(zone_id, zone_count, level)

                if config["alert_settings"]["enable_sound"]:
                    try:
                        winsound.Beep(
                            config["alert_settings"].get("beep_frequency", 1000),
                            config["alert_settings"].get("beep_duration", 500)
                        )
                    except Exception:
                        pass

        else:
            alerted_zones.discard(zone_id)

            last_alert_state[zone_id] = {
                "level": level,
                "count": zone_count
            }
        # (Zone drawing moved to overlay stage to avoid double rendering)
    global heatmap_acc
    if heatmap_acc is None or heatmap_acc.width != width or heatmap_acc.height != height:
        heatmap_acc = zutil.HeatmapAccumulator(width, height)
    heatmap_acc.add_points(centroids, width, height)
    with state_lock:
        current_zone_data = frame_data
    try:
        with open(os.path.join(BASE_DIR, "zone_data.json"), "w") as f:
            json.dump(frame_data, f)
    except Exception:
        pass
    # Remove duplicate total text drawing here; _draw_grid_overlay will add it
    return frame_out, frame_data


# Helper: always draw zone grid overlay using latest data
def _draw_grid_overlay(frame, frame_data):
    try:
        if zone_config:
            runtime_zones_draw = zutil.get_runtime_zones(zone_config, frame.shape[1], frame.shape[0])
            zone_levels = {z["id"]: (z["count"], z["level"]) for z in frame_data.get("zones", [])}
            zutil.draw_zone_polygons(frame, runtime_zones_draw, zone_levels)
            total = int((frame_data or {}).get("total", 0))
            cv2.putText(frame, f"Total People: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            for (x1, y1, x2, y2, conf) in latest_boxes:
                color = (255, 0, 255) if SIMPLE_MODE else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            return
        grid_size = config["detection_settings"]["grid_size"]

        rows, cols = grid_size["rows"], grid_size["cols"]
        h, w = frame.shape[:2]
        zone_h, zone_w = h // rows, w // cols
        # build a map from id->(count, level)
        zone_map = {}
        for z in (frame_data or {}).get("zones", []):
            zone_map[z.get("id")] = (int(z.get("count", 0)), z.get("level", "Low"))
        def color_for(level):
            return (0, 255, 0) if level == "Low" else (0, 255, 255) if level == "Medium" else (0, 165, 255) if level == "High" else (0, 0, 255)
        for r in range(rows):
            for c in range(cols):
                zone_id = f"Z{r * cols + c + 1}"
                count, level = zone_map.get(zone_id, (0, "Low"))
                x0, y0 = c * zone_w, r * zone_h
                x1, y1 = x0 + zone_w, y0 + zone_h
                cv2.rectangle(frame, (x0, y0), (x1, y1), color_for(level), 2)
                cv2.putText(frame, f"{level} ({count})", (x0 + 5, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_for(level), 2)
        # Total label if present
        total = int((frame_data or {}).get("total", 0))
        cv2.putText(frame, f"Total People: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if ADAPTIVE_ENABLED:
            try:
                cv2.putText(frame, f"conf={adaptive_conf:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            except Exception:
                pass
        # Draw person boxes last so they sit on top of grid
        for (x1, y1, x2, y2, conf) in latest_boxes:
            color = (255, 0, 255) if SIMPLE_MODE else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if DEBUG_DETECTION:
                try:
                    cv2.putText(frame, f"{conf:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                except Exception:
                    pass
    except Exception:
        pass


def generate_frames():
    global is_streaming, last_frame_jpeg, latest_frame, latest_heatmap
    # PURE_SIMPLE diagnostic path: bypass all advanced logic to isolate issues
    if PURE_SIMPLE:
        frame_idx = 0
        print("🧪 PURE_SIMPLE mode active: using ultra-minimal detection loop")
        while is_streaming:
            with _capture_lock:
                if cap is None or not cap.isOpened():
                    if not initialize_camera():
                        time.sleep(0.5)
                        continue
                ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            frame_idx += 1
            try:
                # Direct model call (autoselects predict internally) without forcing imgsz; default conf
                results = model(frame)
            except Exception as e:
                print(f"❌ PURE_SIMPLE inference error: {e}")
                results = []
            people = 0
            boxes_local = []
            for r in results:
                for box in getattr(r, 'boxes', []) or []:
                    try:
                        cls = int(box.cls[0])
                    except Exception:
                        cls = -1
                    if cls != 0:
                        continue
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0]) if getattr(box, 'conf', None) is not None else 0.0
                        people += 1
                        boxes_local.append((x1, y1, x2, y2, conf))
                    except Exception:
                        pass
            # Draw
            for (x1, y1, x2, y2, conf) in boxes_local:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(frame, f"People: {people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if frame_idx % 30 == 0:
                confs = [f"{b[4]:.2f}" for b in boxes_local]
                print(f"[PURE_SIMPLE] Frame {frame_idx} people={people} confidences={confs}")
            try:
                ok_j, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok_j:
                    frame_bytes = buffer.tobytes()
                    _last_frame_jpeg = frame_bytes
                else:
                    frame_bytes = _last_frame_jpeg or _ensure_placeholder_frame()
            except Exception:
                frame_bytes = _last_frame_jpeg or _ensure_placeholder_frame()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.02)
        return
    # Start YOLO worker thread if not already running
    if not hasattr(generate_frames, 'worker_started') or not generate_frames.worker_started:
        t = threading.Thread(target=yolo_worker, daemon=True)
        t.start()
        generate_frames.worker_started = True
    try:
        while is_streaming:
            try:
                frame, _ = process_frame()
            except Exception as e:
                print(f"⚠️ process_frame error: {e}")
                frame = None
            if frame is not None:
                with state_lock:
                    snapshot = dict(current_zone_data) if isinstance(current_zone_data, dict) else {"total":0, "zones":[]}
                _draw_grid_overlay(frame, snapshot)
                latest_frame = frame.copy()
                if heatmap_acc is not None:
                    heatmap_frame = np.zeros_like(frame)
                    heatmap_frame[:] = (128, 0, 0)   # Blue background (BGR)

                    latest_heatmap = heatmap_acc.render(heatmap_frame)
            frame_bytes = None
            if frame is not None:
                try:
                    latest_frame = frame.copy()# Faster JPEG encode with lower quality
                    ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok:
                        frame_bytes = buffer.tobytes()
                        _last_frame_jpeg = frame_bytes
                except Exception as e:
                    print(f"⚠️ encode error: {e}")
            if frame_bytes is None:
                frame_bytes = _last_frame_jpeg or _ensure_placeholder_frame()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.02)
    finally:
        pass

def generate_heatmap_frames():
    global is_streaming

    while is_streaming:
        frame, frame_data = process_frame()

        if frame is not None and heatmap_acc is not None:

            # Create a blue background instead of using the camera image
            heatmap_frame = np.zeros_like(frame)
            heatmap_frame[:] = (128, 0, 0)   # Blue (BGR)

            # Draw only the heatmap on the blue background
            frame = heatmap_acc.render(heatmap_frame)

        if frame is not None:
            ok, buf = cv2.imencode(
                '.jpg',
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )

            if ok:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + buf.tobytes()
                    + b'\r\n'
                )

        time.sleep(0.03)

@app.route('/heatmap_feed')
@login_required
def heatmap_feed():
    global is_streaming
    if not is_streaming:
        is_streaming = True
    return Response(generate_heatmap_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
@login_required
def video_feed():
    global is_streaming
    # Always provide a stream; if camera is inactive we'll send placeholder frames to avoid resets
    if not is_streaming:
        is_streaming = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/diagnostic_feed')
def diagnostic_feed():
    """Unauthenticated minimal feed for quick local diagnosis when detection fails.
    Enable by setting pure_simple=true in detection_settings. Access only on localhost recommended."""
    global is_streaming
    if not PURE_SIMPLE:
        return jsonify({"error": "Enable pure_simple in config to use diagnostic feed"}), 400
    if not is_streaming:
        initialize_camera()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/zones')
def get_zones():
    with state_lock:
        return jsonify(current_zone_data)

@app.route('/api/start')
@login_required
def start_camera_api():
    if camera_active:
        return jsonify({"status": "already_active", "message": "Camera is already active and monitoring"})
    
    if initialize_camera():
        return jsonify({"status": "started", "message": "Camera started and monitoring began successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to initialize camera"}), 500

@app.route('/api/stop')
@login_required
def stop_camera_api():
    if not camera_active:
        return jsonify({"status": "already_stopped", "message": "Camera is already stopped"})
    
    stop_camera()
    return jsonify({"status": "stopped", "message": "Camera stopped and monitoring ended"})

@app.route('/api/alerts')
@login_required
def get_alerts():
    try:
        limit = int(request.args.get('limit', 3))
    except Exception:
        limit = 3
    limit = max(1, min(limit, 50))
    with state_lock:
        recent = list(alerts_log)[-limit:][::-1]
    return jsonify({"alerts": recent})

@app.route('/api/status')
@login_required
def get_status():
    return jsonify({
        "camera_active": camera_active,
        "is_streaming": is_streaming,
        "camera_initialized": cap is not None and cap.isOpened(),
        "total_zones": len(current_zone_data.get("zones", [])),
        "alerted_zones": list(alerted_zones)
    })

# =====================
# Video Upload & Analyze
# =====================

ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}

def allowed_video(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_VIDEO_EXTS


@app.route('/downloads/<path:filename>')
@login_required
def downloads(filename):
    # Serve processed files (videos/reports)
    return send_from_directory(DOWNLOADS_DIR, filename, as_attachment=False)


@app.route('/upload', methods=['GET', 'POST'])
def video_upload():
    if request.method == 'GET':
        # If session token param present, render page in viewer mode
        token = request.args.get('session')
        if token:
            with upload_lock:
                sess = upload_sessions.get(token)
            if not sess:
                flash('Upload session not found or finished.', 'error')
                return render_template('upload.html')
            return render_template('upload.html', session_token=token, feed_url=url_for('uploaded_feed', token=token))
        return render_template('upload.html')

    # POST: handle video upload and process
    file = request.files.get('video')
    if not file or file.filename == '':
        flash('Please choose a video file to upload.', 'error')
        return redirect(url_for('video_upload'))

    if not allowed_video(file.filename):
        flash('Unsupported file type. Please upload a video (mp4, avi, mov, mkv, wmv).', 'error')
        return redirect(url_for('video_upload'))

    safe_name = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    upload_path = os.path.join(UPLOADS_DIR, f"{timestamp}_{safe_name}")
    file.save(upload_path)

    # Start a live processing session and show stream immediately
    token = uuid.uuid4().hex
    with upload_lock:
        upload_sessions[token] = {
            'path': upload_path,
            'metrics': {
                'fps': 0.0,
                'frame_width': 0,
                'frame_height': 0,
                'total_frames': 0,
                'latest_people': 0,
                'avg_people': 0.0,
                'max_people': 0,
                'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'done': False,
                'error': None,
            },
            'active': True,
            'zone_data': make_default_zone_data(),
        }

    # Render upload page in viewer mode
    return redirect(url_for('video_upload', session=token))


def _generate_uploaded_frames(token: str, heatmap: bool = False):
    """MJPEG generator for uploaded video sessions."""
    with upload_lock:
        sess = upload_sessions.get(token)
    if not sess:
        return
    video_path = sess['path']
    cap_u = cv2.VideoCapture(video_path)
    if not cap_u.isOpened():
        with upload_lock:
            sess['metrics']['error'] = 'Failed to open the uploaded video.'
            sess['active'] = False
        return

    fps = cap_u.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps if fps and fps > 0 else 25.0
    delay = 1.0 / float(fps)
    frame_w = int(cap_u.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_h = int(cap_u.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    people_counts = []
    total_frames = 0
    upload_heatmap = zutil.HeatmapAccumulator(frame_w, frame_h) if heatmap else None

    with upload_lock:
        sess['metrics']['fps'] = fps
        sess['metrics']['frame_width'] = frame_w
        sess['metrics']['frame_height'] = frame_h

    try:
        while True:
            ok, frame = cap_u.read()
            if not ok or frame is None:
                with upload_lock:
                    sess['metrics']['done'] = True
                    sess['active'] = False
                break

            # Run crowd-friendly upload detection. Tiling improves small-person detection in wide crowd videos.
            unique_boxes, detect_error = _detect_people_upload(frame)
            if detect_error:
                with upload_lock:
                    sess['metrics']['error'] = detect_error
            count_people = len(unique_boxes)
            centroids = []
            # Build zone data for upload (same grid config)
            try:
                grid_size = config["detection_settings"]["grid_size"]
                rows, cols = grid_size["rows"], grid_size["cols"]
            except Exception:
                rows, cols = 3, 3
            h, w = frame.shape[:2]
            zone_h, zone_w = h // rows, w // cols
            zone_counts = [[0 for _ in range(cols)] for _ in range(rows)]
            for (x1, y1, x2, y2, conf) in unique_boxes:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                centroids.append((cx, cy))
                if not heatmap:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"person {conf:.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                r = min(max(cy // zone_h, 0), rows-1)
                c = min(max(cx // zone_w, 0), cols-1)
                zone_counts[r][c] += 1
            thresholds = config["zone_thresholds"]
            upload_zone_data = {"total": int(count_people), "zones": []}
            for r in range(rows):
                for c in range(cols):
                    zid = f"Z{r*cols + c + 1}"
                    zc = zone_counts[r][c]
                    if zc <= 5:
                        level = 'Low'
                    elif zc <= 10:
                        level = 'Medium'
                    elif zc <= 15:
                        level = 'High'
                    else:
                        level = 'Critical'
                    upload_zone_data['zones'].append({"id": zid, "count": zc, "level": level})
                    if level == 'Critical':
                        with state_lock:
                            if zid not in alerted_zones:
                                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                msg = f"[{ts}] ALERT: {zid} is in CRITICAL state with {zc} people"
                                alerts_log.append({
                                    "timestamp": ts,
                                    "zone": zid,
                                    "level": "CRITICAL",
                                    "count": zc,
                                    "message": msg
                                })
                                alerted_zones.add(zid)
                                last_alert_state[zid] = {"level": "Critical", "count": zc}
                                send_alert_email(zid, zc, level)
                    else:
                        with state_lock:
                            if zid in alerted_zones:
                                alerted_zones.remove(zid)
                            last_alert_state[zid] = {"level": level, "count": zc}
            with upload_lock:
                if token in upload_sessions:
                    upload_sessions[token]['zone_data'] = upload_zone_data
            # Overlay
            if heatmap and upload_heatmap is not None:
                upload_heatmap.add_points(centroids, w, h)
                heatmap_frame = np.zeros_like(frame)
                heatmap_frame[:] = (128, 0, 0)
                frame = upload_heatmap.render(heatmap_frame)

            cv2.putText(frame, f"Total People: {count_people}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 3)

            people_counts.append(int(count_people))
            total_frames += 1

            avg_people = (sum(people_counts) / total_frames) if total_frames else 0
            max_people = max(people_counts) if people_counts else 0

            with upload_lock:
                if token in upload_sessions:
                    upload_sessions[token]['metrics'].update({
                        'total_frames': total_frames,
                        'latest_people': int(count_people),
                        'avg_people': round(avg_people, 2),
                        'max_people': int(max_people),
                    })

            # Encode and yield
            ok_j, buffer = cv2.imencode('.jpg', frame)
            if ok_j:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(delay)
    finally:
        try:
            cap_u.release()
        except Exception:
            pass
    # Do NOT remove session here; keep data accessible for zone queries until explicit cleanup policy


@app.route('/uploaded_feed/<token>')
def uploaded_feed(token: str):
    return Response(_generate_uploaded_frames(token), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/uploaded_heatmap_feed/<token>')
def uploaded_heatmap_feed(token: str):
    return Response(_generate_uploaded_frames(token, heatmap=True), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/upload_metrics/<token>')
@login_required
def upload_metrics(token: str):
    with upload_lock:
        sess = upload_sessions.get(token)
        data = sess['metrics'] if sess else {'error': 'not_found'}
    return jsonify(data)


@app.route('/video-upload')
def upload_alias():
    # Friendly alias used by the provided HTML navbar
    return redirect(url_for('video_upload'))


@app.route('/api/upload-video', methods=['POST'])
def api_upload_video():
    file = request.files.get('video')
    if not file or file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400
    if not allowed_video(file.filename):
        return jsonify({'status': 'error', 'message': 'Unsupported file type'}), 400
    safe_name = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    upload_path = os.path.join(UPLOADS_DIR, f"{timestamp}_{safe_name}")
    try:
        file.save(upload_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to save file: {e}'}), 500
    # Create a new upload session identical to /upload route
    token = uuid.uuid4().hex
    with upload_lock:
        upload_sessions[token] = {
            'path': upload_path,
            'metrics': {
                'fps': 0.0,
                'frame_width': 0,
                'frame_height': 0,
                'total_frames': 0,
                'latest_people': 0,
                'avg_people': 0.0,
                'max_people': 0,
                'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'done': False,
                'error': None,
            },
            'active': True,
            'zone_data': make_default_zone_data(),
        }
    return jsonify({'status': 'success', 'filepath': upload_path, 'filename': safe_name, 'session_token': token})


def _analyze_video_background(video_path: str, token: str):
    global upload_analysis_active, upload_analysis_stop
    cap_u = None
    try:
        cap_u = cv2.VideoCapture(video_path)
        if not cap_u.isOpened():
            upload_analysis_active = False
            return
        # Read FPS and consider frame skipping to speed up on CPU
        fps = cap_u.get(cv2.CAP_PROP_FPS) or 25.0
        fps = fps if fps and fps > 0 else 25.0
        frame_skip = 0
        if fps > 25:
            frame_skip = int(fps // 25) - 1  # aim ~25 FPS processing equivalent
            frame_skip = max(0, frame_skip)

        grid_size = config["detection_settings"]["grid_size"]
        rows, cols = grid_size["rows"], grid_size["cols"]
        total_frames = 0

        while not upload_analysis_stop:
            ok, frame = cap_u.read()
            if not ok or frame is None:
                break

            # Skip frames if needed
            if frame_skip > 0:
                for _ in range(frame_skip):
                    cap_u.read()

            unique_boxes, detect_error = _detect_people_upload(frame)
            if detect_error:
                unique_boxes = []
            total_count = len(unique_boxes)
            for (x1, y1, x2, y2, conf) in unique_boxes:
                try:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    row = min(cy // zone_h, rows - 1)
                    col = min(cx // zone_w, cols - 1)
                    zone_counts[row][col] += 1
                except Exception:
                    pass

            frame_data = {"total": int(total_count), "zones": []}
            thresholds = config["zone_thresholds"]
            for r_i in range(rows):
                for c_i in range(cols):
                    zone_id = f"Z{r_i * cols + c_i + 1}"
                    zone_count = int(zone_counts[r_i][c_i])
                    if zone_count <= 5:
                        level = "Low"
                    elif zone_count <= 10:
                        level = "Medium"
                    elif zone_count <= 15:
                        level = "High"
                    else:
                        level = "Critical"
                    frame_data["zones"].append({"id": zone_id, "count": zone_count, "level": level})

                    # Emit alerts similar to live camera so monitoring reflects uploads too
                    if level == "Critical":
                        with state_lock:
                            if zone_id not in alerted_zones:
                                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                msg = f"[{ts}] ALERT: {zone_id} is in CRITICAL state with {zone_count} people"
                                alerts_log.append({
                                    "timestamp": ts,
                                    "zone": zone_id,
                                    "level": "CRITICAL",
                                    "count": zone_count,
                                    "message": msg
                                })
                                alerted_zones.add(zone_id)
                                last_alert_state[zone_id] = {"level": "Critical", "count": zone_count}
                                send_alert_email(zone_id, zone_count, level)
                            else:
                                prev = last_alert_state.get(zone_id)
                                if (not prev) or (prev.get("count") != zone_count):
                                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    msg = f"[{ts}] ALERT: {zone_id} is in CRITICAL state with {zone_count} people"
                                    alerts_log.append({
                                        "timestamp": ts,
                                        "zone": zone_id,
                                        "level": "CRITICAL",
                                        "count": zone_count,
                                        "message": msg
                                    })
                                    last_alert_state[zone_id] = {"level": "Critical", "count": zone_count}
                    else:
                        with state_lock:
                            if zone_id in alerted_zones:
                                alerted_zones.remove(zone_id)
                            last_alert_state[zone_id] = {"level": level, "count": zone_count}

            # Store per-session zone data instead of overriding live monitoring
            with upload_lock:
                if token in upload_sessions:
                    upload_sessions[token]['zone_data'] = frame_data

            # Small sleep to avoid CPU spikes when skipping many frames
            time.sleep(0.001)
            total_frames += 1
    finally:
        if cap_u is not None:
            try:
                cap_u.release()
            except Exception:
                pass
        upload_analysis_active = False
        upload_analysis_stop = False


@app.route('/api/process-video', methods=['POST'])
def api_process_video():
    global upload_analysis_active, upload_analysis_stop, upload_analysis_thread, upload_analysis_path
    try:
        data = request.get_json(force=True)
    except Exception:
        data = {}
    video_path = (data or {}).get('video_path')
    token = (data or {}).get('session_token')
    if not video_path or not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': 'Invalid video path'}), 400
    # Security: ensure it's within our uploads directory
    if not os.path.abspath(video_path).startswith(os.path.abspath(UPLOADS_DIR)):
        return jsonify({'status': 'error', 'message': 'Path not allowed'}), 400

    # Resolve/validate session token; allow fallback by video_path match
    resolved = False
    if token and token in upload_sessions:
        resolved = True
    # Fallback 1: match by exact video_path
    if not resolved and video_path:
        with upload_lock:
            for tk, sess in upload_sessions.items():
                if sess.get('path') == video_path:
                    token = tk
                    resolved = True
                    break
    # Fallback 2: if only one session exists, use it
    if not resolved:
        with upload_lock:
            if len(upload_sessions) == 1:
                token = next(iter(upload_sessions.keys()))
                resolved = True
    # Fallback 3: match by basename of video file
    if not resolved and video_path:
        base = os.path.basename(video_path)
        with upload_lock:
            for tk, sess in upload_sessions.items():
                if os.path.basename(sess.get('path','')) == base:
                    token = tk
                    resolved = True
                    break
    if not resolved:
        # As last resort: auto-create a session for this video_path if file exists
        if video_path and os.path.exists(video_path):
            token = uuid.uuid4().hex
            with upload_lock:
                upload_sessions[token] = {
                    'path': video_path,
                    'metrics': {
                        'fps': 0.0,
                        'frame_width': 0,
                        'frame_height': 0,
                        'total_frames': 0,
                        'latest_people': 0,
                        'avg_people': 0.0,
                        'max_people': 0,
                        'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'done': False,
                        'error': None,
                    },
                    'active': True,
                    'zone_data': make_default_zone_data(),
                }
            resolved = True
            print(f"🆕 process-video: auto-created session token={token} for path={video_path}")
        else:
            print(f"❌ process-video: unresolved session token. Provided token={token} video_path={video_path} active_sessions={list(upload_sessions.keys())}")
            return jsonify({'status': 'error', 'message': 'Invalid or missing session token'}), 400
    else:
        print(f"▶️ process-video: using session token={token} path={video_path}")

    # If a previous analysis is running (any), stop it (single analysis policy)
    if upload_analysis_active and upload_analysis_thread and upload_analysis_thread.is_alive():
        upload_analysis_stop = True
        try:
            upload_analysis_thread.join(timeout=2.0)
        except Exception:
            pass
        upload_analysis_active = False

    upload_analysis_stop = False
    upload_analysis_path = video_path
    upload_analysis_active = True
    upload_analysis_thread = threading.Thread(target=_analyze_video_background, args=(video_path, token), daemon=True)
    upload_analysis_thread.start()

    return jsonify({'status': 'success', 'session_token': token})

@app.route('/api/debug/upload-sessions')
def debug_upload_sessions():
    summary = {}
    with upload_lock:
        for tk, sess in upload_sessions.items():
            summary[tk] = {
                'path': sess.get('path'),
                'has_zone_data': 'zone_data' in sess,
                'active': sess.get('active'),
                'frames': sess.get('metrics', {}).get('total_frames')
            }
    return jsonify(summary)

@app.route('/api/upload_zones/<token>')
def api_upload_zones(token: str):
    with upload_lock:
        sess = upload_sessions.get(token)
        if not sess:
            return jsonify({'total':0, 'zones': []})
        data = sess.get('zone_data') or make_default_zone_data()
    return jsonify(data)

# Default zones helper so UI always shows a grid
def make_default_zone_data():
    try:
        grid_size = config["detection_settings"]["grid_size"]
        rows, cols = grid_size.get("rows", 3), grid_size.get("cols", 3)
    except Exception:
        rows, cols = 3, 3
    data = {"total": 0, "zones": []}
    for r in range(rows):
        for c in range(cols):
            zone_id = f"Z{r * cols + c + 1}"
            data["zones"].append({"id": zone_id, "count": 0, "level": "Low"})
    return data

# Ensure an initial grid is available
try:
    current_zone_data = make_default_zone_data()
except Exception:
    pass

if __name__ == '__main__':
    init_db()
    srv = config.get('server_settings', {})
    host = srv.get('host', '0.0.0.0')
    port = int(srv.get('port', 5000))
    debug = bool(srv.get('debug', False))
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
