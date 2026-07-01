import json
import time
import base64
from datetime import datetime, timezone, timedelta
import os

last_voice_message = ""

# Malaysia Time (UTC+8, no DST) — keeps Pi telemetry stamps in step with the
# Streamlit dashboard, which also uses MYT.
MYT = timezone(timedelta(hours=8), name="MYT")

import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARDWARE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import cv2
from ultralytics import YOLO
import easyocr
from gpiozero import DistanceSensor, TonalBuzzer
from gpiozero.tones import Tone
from gpiozero.pins.pigpio import PiGPIOFactory
import serial
import pynmea2

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MQTT_BROKER          = "broker.hivemq.com"
MQTT_PORT            = 1883
MQTT_TOPIC           = "sdas/pi5/01/telemetry"
MQTT_TOPIC_FRAME     = "sdas/pi5/01/frame"   # annotated JPEG frames (base64)
PUBLISH_INTERVAL_SEC = 1.0

# JPEG encode quality for the frame stream (1-100).
# Lower = smaller payload, faster MQTT. 50 is fine for a dashboard preview.
FRAME_JPEG_QUALITY   = 50

MODEL_PATH           = "/home/wcc/SDAS/road_detection_signs.pt"
GPS_PORT             = "/dev/serial0"
GPS_BAUDRATE         = 9600

ULTRASONIC_ECHO      = 23
ULTRASONIC_TRIGGER   = 18
ULTRASONIC_MAX_M     = 4.0

BUZZER_PIN           = 22

# YOLO class ID for speed limit signs (update if your model differs)
SPEED_LIMIT_CLASS_ID = 47

# Minimum YOLO confidence to publish a detection.
# Model trained at 87% precision / 81% mAP50.
# Blank-frame hallucination threshold is ~0.51 so 0.55 filters those out.
CONFIDENCE_THRESHOLD = 0.55

# Reject frames where mean brightness is below this (0-255).
# Real lit frames are typically >30; dark/blank frames are near 0.
MIN_FRAME_BRIGHTNESS = 20

# ─────────────────────────────────────────────────────────────────────────────
# 3. HARDWARE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("Initializing sensors...")

# YOLO model + webcam
model = YOLO(MODEL_PATH, task="detect")
print(f"[CAMERA] YOLO model loaded. Classes: {list(model.names.values())}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("[CAMERA] ERROR: Cannot open webcam (index 0). Check USB connection.")

# Warm-up: discard frames until the webcam auto-exposure stabilises.
# The model hallucinates on dark frames, so we keep reading until the frame
# is bright enough rather than just counting a fixed number.
print("[CAMERA] Waiting for webcam exposure to stabilise...")
for _ in range(30):          # read up to 30 frames (about 1 second at 30fps)
    ok, _f = cap.read()
    if ok and _f is not None and _f.mean() > MIN_FRAME_BRIGHTNESS:
        break
    time.sleep(0.03)
print(f"[CAMERA] Webcam opened OK. Resolution: "
      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# EasyOCR: Pi 5 has no CUDA GPU so gpu=False is correct here
reader = easyocr.Reader(["en"], gpu=False)
print("[CAMERA] EasyOCR reader ready.")

# HC-SR04 ultrasonic + buzzer
# PiGPIOFactory gives accurate timing. Requires: sudo pigpiod
# Falls back to software PWM silently if pigpiod is not running.
import io, contextlib, warnings
from gpiozero.exc import DistanceSensorNoEcho
try:
    _suppress = io.StringIO()
    with contextlib.redirect_stderr(_suppress):
        _pin_factory = PiGPIOFactory()
    print("[GPIO] pigpiod connected -- accurate ultrasonic timing enabled.")
except Exception:
    _pin_factory = None
    print("[GPIO] pigpiod not running -- using software PWM (less accurate).")
    print("       Run 'sudo pigpiod' before starting this script for best results.")

sensor = DistanceSensor(
    echo=ULTRASONIC_ECHO,
    trigger=ULTRASONIC_TRIGGER,
    max_distance=ULTRASONIC_MAX_M,
    pin_factory=_pin_factory,
)
buzzer = TonalBuzzer(BUZZER_PIN, pin_factory=_pin_factory)

# NEO-6M GPS
gps_port = serial.Serial(GPS_PORT, baudrate=GPS_BAUDRATE, timeout=1)

print("All sensors initialized.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SENSOR FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# --- Persistent GPS state (updated each loop) ---
_gps_state = {"speed_kmh": 0.0, "lat": 0.0, "lon": 0.0}


def read_camera():
    """
    Captures one frame, runs YOLO + EasyOCR, draws bounding boxes on the
    frame, shows it in a local cv2 window (Pi monitor), and returns both
    the detection data and the annotated frame as a base64 JPEG string for
    publishing to the Streamlit dashboard via MQTT.
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[CAMERA] WARNING: cap.read() returned no frame.")
        return {"road_sign": "NONE", "confidence": 0.0, "ocr_text": "", "frame_b64": ""}

    brightness = frame.mean()
    if brightness < MIN_FRAME_BRIGHTNESS:
        print(f"[CAMERA] Frame too dark (brightness={brightness:.1f}) -- skipping.")
        _show_frame(frame)
        return {"road_sign": "NONE", "confidence": 0.0, "ocr_text": "", "frame_b64": _encode_frame(frame)}

    best_sign = "NONE"
    best_conf = 0.0
    best_ocr  = ""

    results = model(frame, stream=True, conf=CONFIDENCE_THRESHOLD,
                    imgsz=640, iou=0.7, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id        = int(box.cls[0])
            conf          = float(box.conf[0])
            label         = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print(f"[CAMERA] brightness={brightness:.1f}  "
                  f"class={label}  conf={conf:.3f}  box=[{x1},{y1},{x2},{y2}]")

            # Draw bounding box and label on frame
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if conf > best_conf:
                best_conf = round(conf, 3)
                best_sign = label

            # OCR on speed-limit crops
            if cls_id == SPEED_LIMIT_CLASS_ID:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    ocr_results = reader.readtext(crop)
                    print(f"  [OCR] raw: {ocr_results}")
                    for (_, text, _prob) in ocr_results:
                        clean = "".join(c for c in text if c.isdigit())
                        if clean:
                            best_ocr = f"{clean}km/h"
                            # Overlay OCR result below the bounding box
                            cv2.putText(frame, best_ocr, (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                            break

    if best_sign == "NONE":
        print(f"[CAMERA] brightness={brightness:.1f}  boxes=0")

    # Show on Pi monitor
    _show_frame(frame)

    return {
        "road_sign":  best_sign,
        "confidence": best_conf,
        "ocr_text":   best_ocr,
        "frame_b64":  _encode_frame(frame),
    }

# Auto-detect headless mode: if there's no DISPLAY env var set (i.e. SSH with
# no X forwarding), force SHOW_LOCAL_CAMERA off to avoid the Qt/XCB crash.
SHOW_LOCAL_CAMERA = bool(os.environ.get("DISPLAY"))

def _show_frame(frame):
    """Display the annotated frame in a local OpenCV window on the Pi monitor.
    Skipped automatically when running headless over SSH."""
    if not SHOW_LOCAL_CAMERA:
        return
    cv2.imshow("SDAS Camera", frame)
    cv2.waitKey(1)   # non-blocking: 1 ms poll keeps the window responsive


def _encode_frame(frame):
    """JPEG-encode the frame and return it as a base64 string for MQTT."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def read_ultrasonic():
    """
    Reads the HC-SR04, triggers the buzzer based on proximity, and returns
    distance in cm.  Returns -1.0 if no echo is received (object out of range
    or wiring issue) so the publisher loop never crashes.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DistanceSensorNoEcho)
            distance_cm = sensor.distance * 100
    except Exception:
        # DistanceSensorNoEcho: no object in range or wiring fault
        buzzer.stop()
        return {"distance_cm": -1.0}

    if distance_cm < 20:
        buzzer.play(Tone("A5"))
    elif distance_cm < 50:
        buzzer.play(Tone("A4"))
    else:
        buzzer.stop()

    return {"distance_cm": round(distance_cm, 1)}


def read_gps():
    """
    Reads all available NMEA lines from the serial buffer this tick and
    updates the persistent GPS state.  Returns the latest known fix.
    Non-blocking: if no valid sentence arrives, the last known values are
    returned so the publisher loop never stalls.
    """
    try:
        # Drain all bytes currently waiting in the buffer (non-blocking)
        waiting = gps_port.in_waiting
        if waiting:
            raw = gps_port.read(waiting).decode("ascii", errors="replace")
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith(("$GPRMC", "$GNRMC", "$GPGGA", "$GNGGA")):
                    try:
                        msg = pynmea2.parse(line)
                        # GPRMC / GNRMC carry speed in knots + lat/lon
                        if hasattr(msg, "spd_over_grnd") and msg.spd_over_grnd is not None:
                            _gps_state["speed_kmh"] = round(
                                float(msg.spd_over_grnd) * 1.852, 2
                            )
                        if hasattr(msg, "latitude") and msg.latitude:
                            _gps_state["lat"] = round(msg.latitude, 6)
                        if hasattr(msg, "longitude") and msg.longitude:
                            _gps_state["lon"] = round(msg.longitude, 6)
                    except pynmea2.ParseError:
                        pass
    except Exception as e:
        print(f"[GPS] Read error: {e}")

    return dict(_gps_state)

def speak(message):
    global last_voice_message

    # Prevent repeating the same message every second
    if message != last_voice_message:
        print(f"[CO-DRIVER] {message}")
        os.system(f'espeak-ng "{message}"')
        last_voice_message = message

def get_co_driver_advice(sign, speed, distance, limit=None):
    """
    Centralised Co-Driver voice-script logic.
    Covers all 63 SIGN_CATALOG states (62 signs + NONE/idle).

    Priority order (highest wins):
      1. Distance-based collision warnings (override everything)
      2. Speed-limit sign with dynamic speed check (uses OCR `limit` when known)
      3. Sign-specific advisory from the lookup table
      4. Fallback: "Road conditions normal."

    Parameters
    ----------
    sign     : str   YOLO class label (case-insensitive)
    speed    : float current GPS speed in km/h
    distance : float ultrasonic distance in cm
    limit    : int|None  OCR-decoded speed-limit value. When None, falls back
               to the legacy 60 km/h check so OCR misses don't go silent.
    """
    sign = (sign or "").lower()

    # 1. Distance-based warning overrides (trumps all signs)
    if 0 < distance < 20:
        return "Collision risk detected. Brake immediately."
    if 0 < distance < 50:
        return "Obstacle ahead. Reduce speed."

    # 2. Dynamic Speed Limit logic (uses the actual OCR value when available)
    if sign == "speed limit":
        if limit:
            if speed > limit:
                return (f"Speed limit {limit} detected. "
                        f"Current speed {speed:.0f}. Reduce speed.")
            return f"Speed limit {limit} detected."
        # Fallback in case OCR fails to read the digits
        if speed > 60:
            return (f"Speed limit sign detected. "
                    f"Current speed {speed:.0f}. Reduce speed.")
        return "Speed limit sign detected."

    # 3. Complete 62-sign mapping (lowercase keys; matches SIGN_CATALOG)
    specific = {
        # ─── HIGH RISK ────────────────────────────────────────────────
        "stop":                              "Stop sign ahead. Prepare to stop.",
        "no entry":                          "No entry ahead. Do not proceed.",
        "children":                          "School zone ahead. Watch for children.",
        "cow nearby":                        "Livestock on road. Slow down.",
        "flagman ahead":                     "Flagman directing traffic. Obey signals.",
        "level crossing with gates ahead":   "Railway crossing ahead. Prepare to stop.",
        "train gate":                        "Train gate ahead. Stop if closing.",
        "traffic signals ahead":             "Traffic light ahead. Be ready to stop.",
        "obstruction":                       "Obstruction on road. Reduce speed.",
        "road work":                         "Road work ahead. Slow down.",
        "pedestrian crossing opt1":          "Pedestrian crossing. Watch for pedestrians.",
        "zebra crossing":                    "Zebra crossing ahead. Give way to pedestrians.",

        # ─── MEDIUM RISK ──────────────────────────────────────────────
        "bumps":                             "Speed bump. Slow down.",
        "bumps ahead":                       "Bumps ahead. Reduce speed.",
        "camera operation zone":             "Speed camera zone. Watch your speed.",
        "chevron (left)":                    "Sharp turn left ahead.",
        "chevron (right)":                   "Sharp turn right ahead.",
        "construction sign":                 "Construction zone. Drive carefully.",
        "crossroad":                         "Crossroad ahead. Watch for crossing traffic.",
        "crossroad on the left":             "Crossroad on left ahead.",
        "crossroad on the right":            "Crossroad on right ahead.",
        "crosswind area":                    "Crosswind area. Grip the wheel firmly.",
        "double bend to left ahead":         "Double bend left ahead. Reduce speed.",
        "double bend to right ahead":        "Double bend right ahead. Reduce speed.",
        "give way":                          "Give way to oncoming traffic.",
        "height limit":                      "Height limit ahead. Check vehicle clearance.",
        "horn prohibited":                   "No horn zone.",
        "left bend ahead":                   "Left bend ahead. Reduce speed.",
        "narrow bridge":                     "Narrow bridge ahead. Drive carefully.",
        "no left turn":                      "Left turn prohibited.",
        "no right turn":                     "Right turn prohibited.",
        "no overtaking":                     "Overtaking prohibited.",
        "no parking":                        "No parking allowed here.",
        "no stopping":                       "No stopping allowed here.",
        "no u-turns":                        "U-turn prohibited.",
        "other dangers nearby":              "Hazard ahead. Stay alert.",
        "pass either side":                  "Pass either side of obstacle.",
        "pass on the left":                  "Keep left to pass.",
        "pass on the right":                 "Keep right to pass.",
        "right bend ahead":                  "Right bend ahead. Reduce speed.",
        "road cones":                        "Road cones ahead. Reduce speed.",
        "road narrows on the left":          "Road narrows on left.",
        "road narrows on the right":         "Road narrows on right.",
        "roadway diverges":                  "Road splits ahead. Choose lane early.",
        "roundabout ahead":                  "Roundabout ahead. Give way.",
        "slippery road":                     "Slippery road. Reduce speed.",
        "t-junction":                        "T-junction ahead. Prepare to turn.",
        "traffic from left merges ahead":    "Traffic merging from left ahead.",
        "traffic from right merges ahead":   "Traffic merging from right ahead.",
        "traffic merging from the left":     "Traffic merging from left.",
        "traffic merging from the right":    "Traffic merging from right.",
        "traffic merging to the left":       "Traffic merging to left.",
        "u turn":                            "U-turn permitted ahead.",
        "weight limit":                      "Weight limit ahead. Check vehicle weight.",

        # ─── LOW RISK ─────────────────────────────────────────────────
        "bicycle lane":                      "Bicycle lane. Share the road.",
        "bus stop":                          "Bus stop ahead.",
        "expressway signs 1":                "Expressway ahead.",
        "expressway signs 2":                "Expressway information ahead.",
        "motorcycles only":                  "Motorcycles only lane.",
        "parking area":                      "Parking area ahead.",
        "towing area":                       "Towing zone.",

        # ─── IDLE ─────────────────────────────────────────────────────
        "none":                              "Road clear.",
    }

    return specific.get(sign, "Road conditions normal.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MQTT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Connected to MQTT Broker: {MQTT_BROKER}")
    else:
        print(f"Failed to connect, reason code {reason_code}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting SDAS Pi 5 Publisher...")

    client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id="pi5-real-publisher")
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    try:
        while True:
            cam_data   = read_camera()
            ultra_data = read_ultrasonic()
            gps_data   = read_gps()

                # ==========================
                # CO-DRIVER ASSISTANT
                # ==========================

            advice = "Road conditions normal"

            sign = cam_data["road_sign"].lower()
            distance = ultra_data["distance_cm"]
            speed = gps_data["speed_kmh"]

            # OCR text looks like "60km/h" — strip non-digits so we pass an int.
            ocr_text = cam_data.get("ocr_text", "") or ""
            ocr_digits = "".join(c for c in ocr_text if c.isdigit())
            limit_value = int(ocr_digits) if ocr_digits else None

            advice = get_co_driver_advice(sign, speed, distance, limit_value)

            speak(advice)

            # Telemetry payload (no frame -- keeps this topic lightweight)
            payload = {
                "timestamp":  datetime.now(MYT).isoformat(timespec="seconds"),
                "camera":     {k: v for k, v in cam_data.items() if k != "frame_b64"},
                "ultrasonic": ultra_data,
                "gps":        gps_data,
                "co_driver": advice
            }
            client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)

            # Frame payload on a separate topic so the dashboard can subscribe
            # independently and the telemetry topic stays small
            if cam_data["frame_b64"]:
                frame_payload = {
                    "timestamp": payload["timestamp"],
                    "frame_b64": cam_data["frame_b64"],
                }
                client.publish(MQTT_TOPIC_FRAME, json.dumps(frame_payload), qos=0)

            print(
                f"Published | sign={cam_data['road_sign']} "
                f"conf={cam_data['confidence']:.2f} "
                f"ocr='{cam_data['ocr_text']}' | "
                f"dist={ultra_data['distance_cm']:.1f}cm | "
                f"speed={gps_data['speed_kmh']:.1f}km/h "
                f"lat={gps_data['lat']} lon={gps_data['lon']}"
            )

            time.sleep(PUBLISH_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nStopping publisher.")
        cap.release()
        cv2.destroyAllWindows()
        buzzer.stop()
        client.loop_stop()
        client.disconnect()