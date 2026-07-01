"""
═══════════════════════════════════════════════════════════════════════════════
  SDAS — Smart Driver Assistance System Dashboard
  SAIA 3353 Machine Learning for IoT — Project 2
  Platform: Raspberry Pi 5  •  Sensors: Camera (YOLO+EasyOCR), Ultrasonic, GPS
  Actuators: Tonal Buzzer, USB Speaker (espeak)
═══════════════════════════════════════════════════════════════════════════════
  Run:   streamlit run app.py
  Deps:  streamlit pandas numpy plotly pydeck paho-mqtt
═══════════════════════════════════════════════════════════════════════════════

  THEME:  "Deep Ocean & Neon" palette — deep midnight blue background with
          dark navy cards and electric-cyan accents. Locked at framework level
          via `.streamlit/config.toml` (toolbarMode = "viewer" hides the
          theme picker).

  MANUAL: A first-time User Manual modal opens on initial load. Users can
          reopen it any time from the "📖 User Manual" button in the sidebar.

  MODES:  Simulation (built-in driving simulator) and MQTT (live) — drains a
          background MQTT client subscribed to telemetry + annotated frames.
          Switching data source auto-clears the logs so sim and live data
          never mix in the same analytics window.
"""

import os
import json
import queue
import random
import time
import base64
from collections import deque
from datetime import datetime, timezone, timedelta

# Malaysia Time (UTC+8, no DST) — used for all wall-clock and telemetry stamps
MYT = timezone(timedelta(hours=8), name="MYT")

def now_myt_iso() -> str:
    """ISO-8601 timestamp in Malaysia time, e.g. '2026-06-19T16:30:12+08:00'."""
    return datetime.now(MYT).isoformat(timespec="seconds")

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG + THEME-AGNOSTIC CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SDAS • Smart Driver Assistance",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── "Deep Ocean & Neon" palette ──
#    Background     #0B132B   Deep Midnight Blue
#    Cards / panels #1C2541   Dark Navy
#    Accent / glow  #00B4D8   Electric Cyan      (#90E0EF Ice Blue for soft hits)
#    Text           #E2EAFC   Soft Frost Blue
#    Muted text     #8B9BC0
CUSTOM_CSS = """
<style>
    /* ── Force the Deep Ocean background everywhere ── */
    .stApp {
        background: linear-gradient(180deg, #0B132B 0%, #0D1F3D 100%) !important;
        color: #E2EAFC;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: #1C2541 !important;
        border-right: 1px solid rgba(0,180,216,0.12);
    }
    [data-testid="stSidebar"] * { color: #E2EAFC; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] .stCaption { color: #8B9BC0; }

    /* ── Body text ── */
    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"] { color: #E2EAFC; }
    .stApp [data-testid="stCaptionContainer"] { color: #8B9BC0; }

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"] { background: #1C2541; }
    [data-testid="stDataFrame"] [role="grid"] { background: #1C2541; color: #E2EAFC; }

    /* ── Code blocks ── */
    [data-testid="stCodeBlock"] pre,
    [data-testid="stCode"] pre,
    pre code { background: #0A1830 !important; color: #E2EAFC !important;
               border: 1px solid rgba(0,180,216,0.18); }

    /* ── Headline ── */
    .stApp .sdas-title,
    [data-testid="stMarkdownContainer"] p.sdas-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        color: #E2EAFC !important;
        margin: 0 !important;
        line-height: 1.15 !important;
    }
    .stApp .sdas-subtitle,
    [data-testid="stMarkdownContainer"] p.sdas-subtitle {
        color: #8B9BC0 !important;
        font-size: 1.05rem !important;
        margin-top: 6px !important;
    }

    /* ── Tabs ── */
    [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab-list"] {
        gap: 25px !important;
        border-bottom: 1px solid rgba(0,180,216,0.18) !important;
    }
    [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 14px 8px !important;
        margin: 0 !important;
    }
    [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p,
    .stTabs [data-baseweb="tab"] p {
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        color: #00B4D8 !important;
        border-bottom: 2px solid #00B4D8 !important;
    }

    /* ── Status banner ── */
    .status-banner {
        padding: 18px 24px; border-radius: 12px; margin: 8px 0 18px 0;
        display: flex; align-items: center; justify-content: space-between;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    .status-SAFE     { background: linear-gradient(135deg,#053744 0%,#00B4D8 120%); }
    .status-WARNING  { background: linear-gradient(135deg,#3D2E0F 0%,#F59E0B 120%); }
    .status-CRITICAL { background: linear-gradient(135deg,#3D0F1A 0%,#EF4444 120%); }
    .status-label   { font-size: 1.6rem; font-weight: 800; color:#FFFFFF; letter-spacing: 0.04em; }
    .status-reason  { color: rgba(255,255,255,0.92); font-size: 0.95rem; }

    /* ── Actuator pill ── */
    .pill { display:inline-block; padding:6px 14px; border-radius:999px; font-weight:600;
            font-size:0.85rem; margin-right:8px; }
    .pill-on  { background:#00B4D8; color:#0B132B; box-shadow: 0 0 14px rgba(0,180,216,0.45); }
    .pill-off { background:#2E3D5C; color:#8B9BC0; }

    /* ── MQTT connection badge ── */
    .conn-badge { display:inline-block; padding:3px 10px; border-radius:999px;
                  font-size:0.78rem; font-weight:600; }
    .conn-up    { background:#053744; color:#00F5D4; border:1px solid #00B4D8; }
    .conn-down  { background:#3D0F1A; color:#FCA5A5; border:1px solid #EF4444; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #1C2541;
        border: 1px solid rgba(0,180,216,0.18);
        border-radius: 10px; padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] { color: #8B9BC0 !important; }
    [data-testid="stMetricValue"] { color: #E2EAFC !important; }

    /* ── Section header ── */
    .section-h { color:#E2EAFC; font-weight:600; font-size:1.05rem;
                 border-bottom:1px solid rgba(0,180,216,0.22);
                 padding-bottom:6px; margin: 14px 0 10px 0; }

    /* ── Info cards ── */
    .info-card {
        background: #1C2541;
        border: 1px solid rgba(0,180,216,0.15);
        border-radius: 10px; padding: 18px; color: #E2EAFC;
        height: 100%;
        transition: border-color .2s ease;
    }
    .info-card:hover { border-color: rgba(0,180,216,0.40); }
    .info-card h4 { margin:0 0 6px 0; color:#E2EAFC; font-size:1.05rem; }
    .info-card p  { margin:0; color:#8B9BC0; font-size:0.9rem; line-height:1.5; }
    .info-card .icon { font-size:1.6rem; margin-bottom:6px; }
    .info-card .codriver-box {
        margin-top:14px; padding:10px; border-radius:6px;
        background: #143E5C; color: #E2EAFC;
        border-left: 3px solid #00B4D8;
    }

    /* ── Sign chip ── */
    .sign-chip {
        display:inline-block; padding:5px 11px; margin:3px;
        border-radius:6px; font-size:0.82rem; font-weight:500;
        border:1px solid rgba(0,180,216,0.18); color:#E2EAFC;
        background:#1C2541;
    }
    .sign-chip.high   { border-left:3px solid #EF4444; }
    .sign-chip.medium { border-left:3px solid #F59E0B; }
    .sign-chip.low    { border-left:3px solid #00B4D8; }

    /* ── Plotly text ── */
    .js-plotly-plot text, .js-plotly-plot .legendtext { fill:#E2EAFC !important; }
    .js-plotly-plot .gtitle { fill:#E2EAFC !important; }

    /* ── Slider accent ── */
    [data-baseweb="slider"] [role="slider"] { background:#00B4D8 !important; }
    [data-baseweb="slider"] div[data-baseweb="slider-track-fill"] { background:#00B4D8 !important; }

    /* ── Buttons ── */
    .stButton > button,
    .stDownloadButton > button {
        background:#1C2541; color:#E2EAFC;
        border:1px solid rgba(0,180,216,0.35);
        transition: all .15s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background:#143E5C; color:#90E0EF;
        border-color:#00B4D8;
        box-shadow: 0 0 12px rgba(0,180,216,0.30);
    }

    /* ── Legacy var(--sdas-*) aliases ── */
    :root {
        --sdas-panel:        #1C2541;
        --sdas-text:         #E2EAFC;
        --sdas-text-muted:   #8B9BC0;
        --sdas-border:       rgba(0,180,216,0.18);
        --sdas-codriver-bg:  #143E5C;
        --sdas-codriver-fg:  #E2EAFC;
    }

    /* ── User-manual modal — softer "Muted Frost" palette ── */
    [role="dialog"] {
        background: #1A2847 !important;
        border: 1px solid rgba(0, 180, 216, 0.20) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55) !important;
    }
    [role="dialog"] h2, [role="dialog"] header, [role="dialog"] [data-testid="stHeading"] { color: #00B4D8 !important; }
    [role="dialog"] h3, [role="dialog"] h4 { color: #C9D8F0 !important; font-weight: 600 !important; }
    [role="dialog"] [data-testid="stMarkdownContainer"] p, [role="dialog"] [data-testid="stMarkdownContainer"] li,
    [role="dialog"] [data-testid="stMarkdownContainer"] td, [role="dialog"] [data-testid="stMarkdownContainer"] th {
        color: #A9BDE0 !important; font-weight: 400 !important;
    }
    [role="dialog"] [data-testid="stMarkdownContainer"] strong, [role="dialog"] [data-testid="stMarkdownContainer"] b {
        color: #D6E4F5 !important;
    }
    [role="dialog"] [data-testid="stMarkdownContainer"] table { border: 1px solid rgba(0, 180, 216, 0.15) !important; }
    [role="dialog"] [data-testid="stMarkdownContainer"] th { background: rgba(0, 180, 216, 0.08) !important; color: #C9D8F0 !important; }
    [role="dialog"] pre, [role="dialog"] code { background: #142139 !important; color: #B8C5DC !important; border: 1px solid rgba(0, 180, 216, 0.12) !important; }
    [role="dialog"] hr { border-color: rgba(0, 180, 216, 0.18) !important; opacity: 0.6; }
    footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIG — thresholds, road sign catalog, MQTT topic plan
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Dashboard distance bands — aligned to the buzzer thresholds hard-coded in
    # pi_publisher.py so the demo is coherent:
    #   < 20 cm  → buzzer A5 (danger)   ↔ dashboard CRITICAL
    #   < 50 cm  → buzzer A4 (caution)  ↔ dashboard WARNING
    "DISTANCE_WARN_CM": 50,
    "DISTANCE_CRIT_CM": 20,
    "SPEED_LIMIT_DEFAULT": 60,
    "ROLLING_WINDOW": 120,
    "REFRESH_SEC": 1.0,
    "MQTT_BROKER": "broker.hivemq.com",
    "MQTT_PORT": 1883,
    "MQTT_TOPIC_BASE": "sdas/pi5/01",
    "LOG_DIR": "logs",
    "AUTOSAVE_SEC": 10,          # min seconds between disk snapshots
}

# Malaysian-context sign catalog with risk weighting.
# Keys MUST match YOLO model class names exactly (case-sensitive).
SIGN_CATALOG = {
    # ─── HIGH RISK ────────────────────────────────────────────────────
    "Stop":                              {"risk": "high",   "action": "Stop the vehicle completely.",                "limit": 0},
    "No entry":                          {"risk": "high",   "action": "Do not enter. Find another route.",           "limit": 0},
    "Children":                          {"risk": "high",   "action": "Children nearby. Drive carefully.",           "limit": 30},
    "Cow nearby":                        {"risk": "high",   "action": "Livestock on road. Slow down.",               "limit": 40},
    "Flagman ahead":                     {"risk": "high",   "action": "Flagman directing traffic. Obey signals.",    "limit": 30},
    "Level crossing with gates ahead":   {"risk": "high",   "action": "Railway crossing ahead. Prepare to stop.",    "limit": 40},
    "Train Gate":                        {"risk": "high",   "action": "Train gate ahead. Stop if closing.",          "limit": 40},
    "Traffic signals ahead":             {"risk": "high",   "action": "Traffic light ahead. Be ready to stop.",      "limit": None},
    "Obstruction":                       {"risk": "high",   "action": "Obstruction on road. Reduce speed.",          "limit": None},
    "Road work":                         {"risk": "high",   "action": "Road work ahead. Slow down.",                 "limit": 40},
    "pedestrian crossing opt1":          {"risk": "high",   "action": "Pedestrian crossing. Watch for pedestrians.", "limit": 30},
    "Zebra crossing":                    {"risk": "high",   "action": "Zebra crossing ahead. Give way to pedestrians.", "limit": 30},

    # ─── MEDIUM RISK ──────────────────────────────────────────────────
    "Bumps":                             {"risk": "medium", "action": "Speed bump. Slow down.",                      "limit": 30},
    "Bumps ahead":                       {"risk": "medium", "action": "Bumps ahead. Reduce speed.",                  "limit": 30},
    "Camera operation zone":             {"risk": "medium", "action": "Speed camera zone. Watch your speed.",        "limit": None},
    "Chevron (left)":                    {"risk": "medium", "action": "Sharp turn left ahead.",                      "limit": None},
    "Chevron (right)":                   {"risk": "medium", "action": "Sharp turn right ahead.",                     "limit": None},
    "Construction sign":                 {"risk": "medium", "action": "Construction zone. Drive carefully.",         "limit": 40},
    "Crossroad":                         {"risk": "medium", "action": "Crossroad ahead. Watch for crossing traffic.", "limit": None},
    "Crossroad on the left":             {"risk": "medium", "action": "Crossroad on left ahead.",                    "limit": None},
    "Crossroad on the right":            {"risk": "medium", "action": "Crossroad on right ahead.",                   "limit": None},
    "Crosswind area":                    {"risk": "medium", "action": "Crosswind area. Grip the wheel firmly.",      "limit": None},
    "Double Bend to Left Ahead":         {"risk": "medium", "action": "Double bend left ahead. Reduce speed.",       "limit": None},
    "Double Bend to Right Ahead":        {"risk": "medium", "action": "Double bend right ahead. Reduce speed.",      "limit": None},
    "Give way":                          {"risk": "medium", "action": "Give way to oncoming traffic.",               "limit": None},
    "Height limit":                      {"risk": "medium", "action": "Height limit ahead. Check vehicle clearance.", "limit": None},
    "Horn Prohibited":                   {"risk": "medium", "action": "No horn zone.",                               "limit": None},
    "Left Bend Ahead":                   {"risk": "medium", "action": "Left bend ahead. Reduce speed.",              "limit": None},
    "Narrow bridge":                     {"risk": "medium", "action": "Narrow bridge ahead. Drive carefully.",       "limit": None},
    "No left turn":                      {"risk": "medium", "action": "Left turn prohibited.",                       "limit": None},
    "No right turn":                     {"risk": "medium", "action": "Right turn prohibited.",                      "limit": None},
    "No overtaking":                     {"risk": "medium", "action": "Overtaking prohibited.",                      "limit": None},
    "No parking":                        {"risk": "medium", "action": "No parking allowed here.",                    "limit": None},
    "No Stopping":                       {"risk": "medium", "action": "No stopping allowed here.",                   "limit": None},
    "No U-turns":                        {"risk": "medium", "action": "U-turn prohibited.",                          "limit": None},
    "Other dangers nearby":              {"risk": "medium", "action": "Hazard ahead. Stay alert.",                   "limit": None},
    "Pass either side":                  {"risk": "medium", "action": "Pass either side of obstacle.",               "limit": None},
    "Pass on the left":                  {"risk": "medium", "action": "Keep left to pass.",                          "limit": None},
    "Pass on the right":                 {"risk": "medium", "action": "Keep right to pass.",                         "limit": None},
    "Right Bend Ahead":                  {"risk": "medium", "action": "Right bend ahead. Reduce speed.",             "limit": None},
    "Road cones":                        {"risk": "medium", "action": "Road cones ahead. Reduce speed.",             "limit": 40},
    "Road narrows on the left":          {"risk": "medium", "action": "Road narrows on left.",                       "limit": None},
    "Road narrows on the right":         {"risk": "medium", "action": "Road narrows on right.",                      "limit": None},
    "Roadway diverges":                  {"risk": "medium", "action": "Road splits ahead. Choose lane early.",       "limit": None},
    "Roundabout ahead":                  {"risk": "medium", "action": "Roundabout ahead. Give way.",                 "limit": None},
    "Slippery road":                     {"risk": "medium", "action": "Slippery road. Reduce speed.",                "limit": 50},
    "Speed limit":                       {"risk": "medium", "action": "Speed limit sign — read OCR value.",          "limit": None},
    "T-junction":                        {"risk": "medium", "action": "T-junction ahead. Prepare to turn.",          "limit": None},
    "Traffic from Left Merges Ahead":    {"risk": "medium", "action": "Traffic merging from left ahead.",            "limit": None},
    "Traffic from Right Merges Ahead":   {"risk": "medium", "action": "Traffic merging from right ahead.",           "limit": None},
    "Traffic merging from the left":     {"risk": "medium", "action": "Traffic merging from left.",                  "limit": None},
    "Traffic merging from the right":    {"risk": "medium", "action": "Traffic merging from right.",                 "limit": None},
    "Traffic merging to the left":       {"risk": "medium", "action": "Traffic merging to left.",                    "limit": None},
    "U turn":                            {"risk": "medium", "action": "U-turn permitted ahead.",                     "limit": None},
    "Weight limit":                      {"risk": "medium", "action": "Weight limit ahead. Check vehicle weight.",   "limit": None},

    # ─── LOW RISK ─────────────────────────────────────────────────────
    "Bicycle lane":                      {"risk": "low",    "action": "Bicycle lane. Share the road.",               "limit": None},
    "Bus stop":                          {"risk": "low",    "action": "Bus stop ahead.",                             "limit": None},
    "Expressway signs 1":                {"risk": "low",    "action": "Expressway ahead.",                           "limit": None},
    "Expressway signs 2":                {"risk": "low",    "action": "Expressway information ahead.",               "limit": None},
    "Motorcycles only":                  {"risk": "low",    "action": "Motorcycles only lane.",                      "limit": None},
    "Parking area":                      {"risk": "low",    "action": "Parking area ahead.",                         "limit": None},
    "Towing area":                       {"risk": "low",    "action": "Towing zone.",                                "limit": None},

    # ─── IDLE ─────────────────────────────────────────────────────────
    "NONE":                              {"risk": "none",   "action": "Road clear.",                                 "limit": None},
}

SPEED_LIMIT_OCR_VALUES = [30, 40, 50, 60, 70, 80, 90, 110]

MQTT_TOPIC_PLAN = """
sdas/pi5/01/telemetry      → full sensor JSON (camera+ultrasonic+gps)
sdas/pi5/01/frame          → annotated JPEG frame, base64 ({frame_b64: ...})
sdas/pi5/01/alerts         → alert events (severity, reason, ts)
sdas/pi5/01/status         → SAFE | WARNING | CRITICAL
sdas/pi5/01/actuator/cmd   → {buzzer:bool}
"""

HARDWARE_SPEC = [
    {"icon": "🧠", "name": "Raspberry Pi 5 (8 GB)",
     "desc": "Main on-board AI processor running the custom YOLO model, EasyOCR, fusion logic and the Streamlit dashboard."},
    {"icon": "📷", "name": "Pi Camera / USB Webcam",
     "desc": "Captures the road scene at 640×480. Frames feed the YOLO model for sign detection and EasyOCR for speed-limit numbers."},
    {"icon": "📏", "name": "HC-SR04 Ultrasonic",
     "desc": "Measures distance to the obstacle ahead (≈2–400 cm) via gpiozero. Drives the collision-warning state machine."},
    {"icon": "🛰️", "name": "NEO-6M GPS",
     "desc": "Provides real-time speed (km/h), latitude and longitude for route logging and over-speed detection."},
    {"icon": "🔔", "name": "Tonal Piezo Buzzer",
     "desc": "Audible alarm via gpiozero TonalBuzzer — escalating tone as the obstacle gets closer / on CRITICAL events."},
    {"icon": "🔊", "name": "USB Mini Speaker",
     "desc": "AI co-driver voice output — converts decisions into natural English instructions via espeak TTS."},
]

CAPABILITIES = [
    ("🚧 Real-time traffic sign recognition",
     "Custom YOLO inference on the Pi 5 identifies 62 Malaysian road sign classes from the live camera feed."),
    ("🔡 EasyOCR speed-limit reading",
     "When a Speed-limit sign is detected, EasyOCR extracts the numeric value (30 / 40 / 50 / 60 / 70 / 80 / 90 / 110)."),
    ("📐 Collision-risk monitoring",
     "Ultrasonic distance is fused with GPS speed. Risk is computed continuously and raised to CRITICAL when impact is imminent."),
    ("🚦 Over-speed detection",
     "GPS speed is compared against the active speed limit. Exceeding by 1+ km/h triggers WARNING; 10+ triggers CRITICAL."),
    ("🗣️ AI Co-Driver voice guidance",
     "Natural English utterances (e.g. “Speed limit 60 detected. Current speed 72. Reduce speed.”) instead of raw sensor labels."),
    ("🗺️ Route logging & map view",
     "GPS fixes are stored in a rolling buffer and rendered as a coloured trajectory on a Mapbox map."),
    ("📊 Live analytics & CSV export",
     "Speed/distance trends, sign frequency, risk surface and downloadable CSV of every sample."),
    ("☁️ MQTT-ready telemetry",
     "Identical JSON schema for simulation and live mode — drop the Pi publisher in and the dashboard works unchanged."),
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. SESSION STATE  (idempotent — safe to call multiple times per run)
# ─────────────────────────────────────────────────────────────────────────────
def ensure_state():
    """Guarantee every session_state key exists. Safe across reruns + iframe
    round-trips that can occasionally drop keys on bleeding-edge Python."""
    defaults = {
        "telemetry":    lambda: deque(maxlen=CONFIG["ROLLING_WINDOW"]),
        "alerts":       lambda: deque(maxlen=200),
        "sign_counts":  dict,
        "running":      lambda: True,
        "latest_frame": lambda: None,   # base64 JPEG from sdas/pi5/01/frame
        "latest_ocr":   lambda: "",     # most recent OCR text from camera
        "prev_source":  lambda: None,   # track Simulation↔MQTT switches
        "last_autosave": lambda: 0.0,   # epoch of last disk snapshot
        "sim_state":    lambda: {
            "speed": 45.0, "distance": 250.0,
            "lat": 3.1390, "lon": 101.6869,
            "heading": 0.0,
            "current_limit": CONFIG["SPEED_LIMIT_DEFAULT"],
        },
    }
    for key, factory in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = factory()

ensure_state()


# ─────────────────────────────────────────────────────────────────────────────
# 4. MQTT CLIENT (BACKGROUND THREAD)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_mqtt():
    """Initializes the MQTT client in a background thread just once per session."""
    q       = queue.Queue()   # telemetry messages
    frame_q = queue.Queue()   # annotated JPEG frames (base64)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if msg.topic.endswith("/frame"):
                frame_q.put(payload)
            else:
                q.put(payload)
        except Exception as e:
            print(f"MQTT Parsing Error: {e}")

    client_id = f"sdas-streamlit-{random.randint(1000,9999)}"
    client = mqtt.Client(client_id=client_id)
    client.on_message = on_message

    try:
        client.connect(CONFIG["MQTT_BROKER"], CONFIG["MQTT_PORT"], 60)
        client.subscribe(f"{CONFIG['MQTT_TOPIC_BASE']}/telemetry", qos=1)
        client.subscribe(f"{CONFIG['MQTT_TOPIC_BASE']}/frame",     qos=0)
        client.loop_start()
    except Exception as e:
        print(f"Failed to start MQTT: {e}")

    return client, q, frame_q

mqtt_client, mqtt_queue, mqtt_frame_queue = init_mqtt()


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIMULATED SENSOR STREAM
# ─────────────────────────────────────────────────────────────────────────────
def generate_sample() -> dict:
    """Realistic driving simulator — produces ~70% SAFE, ~22% WARNING, ~8% CRITICAL
    by aiming the vehicle to drive *under* the limit with a comfortable headway,
    and only occasionally introducing an obstacle or speed spike."""
    s = st.session_state.sim_state

    # ── Speed: aim slightly UNDER the active limit (defensive driver) ──
    target = s["current_limit"] + random.uniform(-10, 5)
    s["speed"] += (target - s["speed"]) * 0.12 + random.uniform(-1.5, 1.5)
    s["speed"] = max(0, min(s["speed"], 140))

    # ── Distance: stay comfortably far most of the time ──
    roll = random.random()
    if roll < 0.02:
        s["distance"] = random.uniform(20, 90)        # CRITICAL obstacle
    elif roll < 0.08:
        s["distance"] = random.uniform(90, 180)       # near-WARNING band
    else:
        s["distance"] += random.uniform(-10, 10)      # gentle drift
        s["distance"] = max(150, min(s["distance"], 400))  # floor at SAFE

    # ── GPS: move forward along heading ──
    s["heading"] += random.uniform(-5, 5)
    delta = s["speed"] / 3600 / 111
    s["lat"] += delta * np.cos(np.radians(s["heading"]))
    s["lon"] += delta * np.sin(np.radians(s["heading"]))

    # ── Sign detection: ~18% of frames see a sign ──
    if random.random() < 0.18:
        sign = random.choice([k for k in SIGN_CATALOG if k != "NONE"])
        conf = round(random.uniform(0.72, 0.98), 2)
        if sign == "Speed limit":
            limit_val = random.choice(SPEED_LIMIT_OCR_VALUES)
            ocr = str(limit_val)
            s["current_limit"] = limit_val
        else:
            ocr = ""
            implied = SIGN_CATALOG[sign]["limit"]
            if implied is not None and implied > 0:
                s["current_limit"] = implied
    else:
        sign, conf, ocr = "NONE", 0.0, ""

    speed_now    = round(s["speed"], 1)
    distance_now = round(s["distance"], 1)

    # Decode the OCR digits into an int so the Co-Driver can name the limit.
    ocr_digits = "".join(c for c in ocr if c.isdigit())
    limit_value = int(ocr_digits) if ocr_digits else None

    return {
        "timestamp": now_myt_iso(),
        "camera":     {"road_sign": sign, "confidence": conf, "ocr_text": ocr},
        "ultrasonic": {"distance_cm": distance_now},
        "gps":        {"speed_kmh": speed_now,
                       "lat": round(s["lat"], 6),
                       "lon": round(s["lon"], 6)},
        # Stamp the Co-Driver line on every simulated sample so evaluate()
        # can read it the same way it reads the Pi publisher's payload.
        "co_driver":  get_co_driver_advice(sign, speed_now, distance_now, limit_value),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def get_co_driver_advice(sign, speed, distance, limit=None):
    """
    Centralised Co-Driver voice-script logic — mirrors pi_publisher.py exactly
    so simulation and live MQTT modes produce identical utterances. Covers all
    63 SIGN_CATALOG states (62 signs + NONE/idle).

    Parameters
    ----------
    sign     : str   YOLO class label (case-insensitive)
    speed    : float current GPS speed in km/h
    distance : float ultrasonic distance in cm
    limit    : int|None  the OCR-decoded speed-limit value when known.
               When None, falls back to the legacy 60 km/h check so the
               function still produces a useful line if OCR misses the digits.
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


def evaluate(sample: dict) -> dict:
    dist  = sample["ultrasonic"]["distance_cm"]
    spd   = sample["gps"]["speed_kmh"]
    sign  = sample["camera"]["road_sign"]
    ocr   = sample["camera"].get("ocr_text", "")
    meta  = SIGN_CATALOG.get(sign, SIGN_CATALOG["NONE"])

    # Pi publisher sends OCR text like "60km/h"; sim sends bare "60".
    # Strip non-digits so both forms parse.
    ocr_digits = "".join(c for c in ocr if c.isdigit())
    if sign == "Speed limit" and ocr_digits:
        limit = int(ocr_digits)
        st.session_state.sim_state["current_limit"] = limit
    elif meta["limit"] is not None and meta["limit"] > 0:
        limit = meta["limit"]
    else:
        limit = st.session_state.sim_state["current_limit"]

    reasons, status = [], "SAFE"

    if dist < CONFIG["DISTANCE_CRIT_CM"]:
        status = "CRITICAL"
        reasons.append(f"Obstacle {dist:.0f} cm — brake immediately")
    elif dist < CONFIG["DISTANCE_WARN_CM"]:
        if status != "CRITICAL":
            status = "WARNING"
        reasons.append(f"Obstacle {dist:.0f} cm — maintain safe distance")

    if limit and spd > limit + 10:
        status = "CRITICAL"
        reasons.append(f"Speed {spd:.0f} km/h exceeds limit {limit} by 10+")
    elif limit and spd > limit:
        if status != "CRITICAL":
            status = "WARNING"
        reasons.append(f"Speed {spd:.0f} km/h over {limit} km/h limit")

    if sign != "NONE" and meta["risk"] == "high":
        if status == "SAFE":
            status = "WARNING"
        reasons.append(f"{sign} detected — {meta['action']}")

    if not reasons:
        reasons.append("All systems nominal")

    actuators = {
        "buzzer": status == "CRITICAL",
    }

    # Voice line is sourced directly from the centralised Co-Driver logic
    # (stamped on the sample by either the Pi publisher or the simulator).
    voice = sample.get("co_driver") or "Road clear."

    return {
        "status": status,
        "reasons": reasons,
        "actuators": actuators,
        "voice": voice,
        "speed_limit": limit,
    }


def push_alert(sample, decision):
    if decision["status"] == "SAFE":
        return
    last = st.session_state.alerts[-1] if st.session_state.alerts else None
    reason = " • ".join(decision["reasons"])
    if last and last["reason"] == reason and last["severity"] == decision["status"]:
        return
    st.session_state.alerts.append({
        "timestamp": sample["timestamp"],
        "severity":  decision["status"],
        "sign":      sample["camera"]["road_sign"],
        "ocr":       sample["camera"].get("ocr_text", "") or "—",
        "speed":     sample["gps"]["speed_kmh"],
        "distance":  sample["ultrasonic"]["distance_cm"],
        "limit":     decision["speed_limit"],
        "reason":    reason,
        "co_driver": decision.get("voice", ""),
    })


def flatten(sample, decision):
    """Single source of truth for the flat telemetry row used by the table + CSV."""
    return {
        "ts":         sample["timestamp"],
        "speed":      sample["gps"]["speed_kmh"],
        "distance":   sample["ultrasonic"]["distance_cm"],
        "sign":       sample["camera"]["road_sign"],
        "confidence": sample["camera"]["confidence"],
        "ocr":        sample["camera"].get("ocr_text", ""),
        "lat":        sample["gps"]["lat"],
        "lon":        sample["gps"]["lon"],
        "status":     decision["status"],
        "limit":      decision["speed_limit"],
        "co_driver":  decision.get("voice", ""),
    }


def autosave_logs():
    """Snapshot the current telemetry + alert buffers to disk, at most once per
    AUTOSAVE_SEC. Overwrites a single per-session file (no per-tick file spam).
    Useful on the Pi for crash recovery; on ephemeral cloud hosts it's a no-op
    once the session ends."""
    now = time.time()
    if now - st.session_state.last_autosave < CONFIG["AUTOSAVE_SEC"]:
        return
    try:
        os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
        if st.session_state.telemetry:
            pd.DataFrame(list(st.session_state.telemetry)).to_csv(
                os.path.join(CONFIG["LOG_DIR"], "telemetry_session.csv"), index=False)
        if st.session_state.alerts:
            pd.DataFrame(list(st.session_state.alerts)).to_csv(
                os.path.join(CONFIG["LOG_DIR"], "alerts_session.csv"), index=False)
        st.session_state.last_autosave = now
    except Exception as e:
        print(f"Autosave failed: {e}")


def style_fig(fig, height=300, margin=(10, 10, 10, 10)):
    """Dark-themed Plotly: dark template + transparent paper for clean blend."""
    l, r, t, b = margin
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=l, r=r, t=t, b=b),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6b. USER MANUAL  (modal popup, first-time auto-show + reopenable from sidebar)
# ─────────────────────────────────────────────────────────────────────────────
@st.dialog("📖 SDAS User Manual", width="medium")
def show_manual():
    """Pop-up modal explaining how to use the dashboard and how decisions are made."""
    st.markdown("""
### Welcome to SDAS

**SDAS** is a real-time IoT dashboard for a Raspberry Pi 5 driver-assistance
system. It fuses a camera (YOLO + EasyOCR), an ultrasonic sensor, and a GPS
module into a single SAFE / WARNING / CRITICAL state, with an AI co-driver that
speaks natural instructions to the driver.

---

### 🚀 Getting started

1. The dashboard auto-streams **simulated** telemetry at ~1 Hz out of the box.
2. The **coloured banner at the top** is the current system state.
3. Use the **sidebar** to switch data source, pause, tune thresholds, clear
   logs, or export the data.
4. Click through the **tabs** to see different views of the same live data.

---

### 🔌 Simulation vs MQTT (live)

| Mode | Where data comes from |
|---|---|
| **Simulation** | Built-in driving simulator — no hardware needed, great for demos. |
| **MQTT (live)** | Subscribes to the Raspberry Pi publisher (`sdas/pi5/01/...`). The **Sign Detection** tab also shows the annotated camera frame. |

> Switching data source **auto-clears the logs** so simulated and live data are
> never mixed in the same analytics window.

---

### 🧭 Tab guide

| Tab | What you'll see |
|---|---|
| **ℹ️ About** | System overview · hardware list · the 62 sign classes |
| **🏠 Overview** | 4 live metric cards + actuator pills + speed/distance trend |
| **📊 Sensors** | Raw rolling-window telemetry table + latest JSON payload |
| **🛑 Sign Detection** | Live camera frame (left) + current YOLO/OCR detection (right) |
| **🚨 Alerts** | Event log of WARNING + CRITICAL events only (incl. OCR value) |
| **🗺️ GPS Map** | Vehicle trajectory on a dark Mapbox map |
| **📈 Analytics** | Speed/distance trends · risk surface · detection frequency |
| **⚙️ Connectivity** | MQTT topic plan + Pi-publisher and subscriber code samples |

---

### 🧠 Decision logic — how the state is decided

Every tick, three **independent** input channels each vote on severity:

```
1.  DISTANCE  (HC-SR04 ultrasonic)
       < 20 cm   → CRITICAL   (matches buzzer tone A5 on the Pi)
       < 50 cm   → WARNING    (matches buzzer tone A4 on the Pi)

2.  SPEED vs LIMIT  (GPS speed + last seen Speed-limit sign)
       > limit + 10  → CRITICAL
       > limit       → WARNING

3.  SIGN RISK  (YOLO classification)
       high-risk sign in frame  → WARNING
       medium / low / no sign   → no contribution
```

**The worst vote wins.** Once any channel says CRITICAL, the others cannot
demote it. SAFE only survives if no channel fires at all.

> **Sidebar thresholds** change the **dashboard banner** in both Simulation
> and MQTT modes. They do **not** change the Pi's physical buzzer — that's
> hard-coded in `pi_publisher.py`. The defaults (20 / 50 cm) are deliberately
> aligned to the buzzer so what you see matches what you hear.

---

### 🗣️ Co-Driver voice script

The Co-Driver line is generated by a **single** function —
`get_co_driver_advice(sign, speed, distance, limit)` — that lives in both
`pi_publisher.py` and `app.py` with identical logic. The Pi stamps it onto
every telemetry packet under the `co_driver` key; the simulator does the
same. The dashboard never re-derives it.

**Priority order (highest wins):**

```
① Distance overrides — strongest signal, ignores the sign
     < 20 cm  →  "Collision risk detected. Brake immediately."
     < 50 cm  →  "Obstacle ahead. Reduce speed."

② Speed-limit sign — dynamic, uses the OCR-decoded limit
     limit known & speed > limit
          →  "Speed limit {limit} detected. Current speed {N}. Reduce speed."
     limit known & speed ≤ limit
          →  "Speed limit {limit} detected."
     limit unknown (OCR miss) & speed > 60
          →  "Speed limit sign detected. Current speed {N}. Reduce speed."
     limit unknown & speed ≤ 60
          →  "Speed limit sign detected."

③ Sign-specific lookup — covers all 62 SIGN_CATALOG classes
     e.g. "Stop"           → "Stop sign ahead. Prepare to stop."
          "Zebra crossing" → "Zebra crossing ahead. Give way to pedestrians."
          "Children"       → "School zone ahead. Watch for children."
          "Slippery road"  → "Slippery road. Reduce speed."

④ Idle  →  "Road clear."   (no sign in frame)
   Unknown sign  →  "Road conditions normal."   (shouldn't happen)
```

All 63 states (62 signs + idle) are mapped — there are no silent fallbacks.
The full script is also shown on the **About** tab for transparency.

---

### 🔔 Actuator mapping

| Final state | Buzzer (Pi) | Co-Driver voice |
|---|---|---|
| **SAFE** | OFF | "Road clear." or sign-specific advisory |
| **WARNING** | A4 tone | Sign-specific advisory or distance warning |
| **CRITICAL** | A5 tone | "Collision risk detected. Brake immediately." |

---

### 📚 Two different logs

| Log | Contains | Visible in |
|---|---|---|
| `telemetry` | **Every** sample (SAFE, WARNING, CRITICAL) | Sensors tab + CSV export |
| `alerts` | Only non-SAFE events, deduped | Alerts tab |

This is intentional: a real ADAS shouldn't fill its event log with "all systems
nominal". Use the **Sensors tab** to prove the monitoring is always live.

---

**Project:** SAIA 3353 Machine Learning for IoT · Project 2
**Platform:** Raspberry Pi 5 (8 GB)
    """)


# Show the manual automatically on the first run of a session
if "manual_seen" not in st.session_state:
    st.session_state.manual_seen = True
    show_manual()


# ─────────────────────────────────────────────────────────────────────────────
# 7. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚗 SDAS Control")
    st.caption("Smart Driver Assistance — Raspberry Pi 5")

    if st.button("📖 User Manual", use_container_width=True):
        show_manual()

    st.markdown("---")

    source = st.radio("Data Source", ["Simulation", "MQTT (live)"], index=0,
                      help="Switch to MQTT when running on the Pi with a live broker")

    # Live MQTT connection badge
    if source == "MQTT (live)":
        connected = False
        try:
            connected = mqtt_client.is_connected()
        except Exception:
            connected = False
        badge = ("<span class='conn-badge conn-up'>● broker connected</span>"
                 if connected else
                 "<span class='conn-badge conn-down'>● broker offline</span>")
        st.markdown(badge, unsafe_allow_html=True)

    st.session_state.running = st.toggle("Stream active", value=st.session_state.running)
    refresh = st.slider("Refresh rate (sec)", 0.5, 3.0, CONFIG["REFRESH_SEC"], 0.1)

    st.markdown("---")
    st.subheader(
        "Thresholds",
        help=(
            "These sliders change the **dashboard banner** "
            "(SAFE / WARNING / CRITICAL) in both Simulation and MQTT "
            "modes.\n\n"
            "They do **not** change the Pi's physical buzzer — that's "
            "hard-coded in `pi_publisher.py` (A5 <20 cm, A4 <50 cm).\n\n"
            "Defaults match the buzzer so the two stay in sync."
        ),
    )
    CONFIG["DISTANCE_WARN_CM"] = st.number_input(
        "Warn distance (cm)", 15, 300, CONFIG["DISTANCE_WARN_CM"],
        help="Obstacle within this distance → dashboard WARNING (default 50 = buzzer A4)",
    )
    crit_max = CONFIG["DISTANCE_WARN_CM"] - 1
    CONFIG["DISTANCE_CRIT_CM"] = st.number_input(
        "Critical distance (cm)", 5, crit_max,
        min(CONFIG["DISTANCE_CRIT_CM"], crit_max),
        help="Obstacle within this distance → dashboard CRITICAL (default 20 = buzzer A5)",
    )

    st.markdown("---")
    autosave = st.toggle("💾 Auto-save logs to disk", value=False,
                         help=f"Snapshot telemetry + alerts to ./{CONFIG['LOG_DIR']}/ "
                              f"every {CONFIG['AUTOSAVE_SEC']}s (useful on the Pi)")

    if st.button("🗑️ Clear logs", use_container_width=True):
        st.session_state.telemetry.clear()
        st.session_state.alerts.clear()
        st.session_state.sign_counts.clear()
        st.rerun()

    if st.session_state.telemetry:
        df_dl = pd.DataFrame(list(st.session_state.telemetry))
        st.download_button(
            "⬇️ Export telemetry CSV",
            df_dl.to_csv(index=False).encode(),
            file_name=f"sdas_telemetry_{int(time.time())}.csv",
            mime="text/csv", use_container_width=True,
        )
    if st.session_state.alerts:
        adf_dl = pd.DataFrame(list(st.session_state.alerts))
        st.download_button(
            "⬇️ Export alerts CSV",
            adf_dl.to_csv(index=False).encode(),
            file_name=f"sdas_alerts_{int(time.time())}.csv",
            mime="text/csv", use_container_width=True,
        )

    st.markdown("---")
    st.caption(f"Samples: {len(st.session_state.telemetry)} | Alerts: {len(st.session_state.alerts)}")
    st.caption("SAIA 3353 · Project 2")


# ─────────────────────────────────────────────────────────────────────────────
# 8. INGEST LIVE STREAM (SIMULATION OR MQTT)
# ─────────────────────────────────────────────────────────────────────────────
ensure_state()

# Auto-clear logs when the data source changes so sim/live data never mix.
if source != st.session_state.prev_source:
    if st.session_state.prev_source is not None:
        st.session_state.telemetry.clear()
        st.session_state.alerts.clear()
        st.session_state.sign_counts.clear()
        st.session_state.latest_frame = None
        st.session_state.latest_ocr = ""
    st.session_state.prev_source = source

sample, flat = None, None
decision = {"status": "SAFE", "reasons": ["Stream paused"],
            "actuators": {"buzzer": False},
            "voice": "Stream paused.", "speed_limit": None}

if st.session_state.running:
    if source == "Simulation":
        sample = generate_sample()
        decision = evaluate(sample)
        flat = flatten(sample, decision)
        st.session_state.telemetry.append(flat)
        if flat["sign"] != "NONE":
            st.session_state.sign_counts[flat["sign"]] = st.session_state.sign_counts.get(flat["sign"], 0) + 1
        push_alert(sample, decision)

    else:  # MQTT (live)
        new_data = False

        # Drain frame queue -- keep only the most recent frame
        while not mqtt_frame_queue.empty():
            try:
                fmsg = mqtt_frame_queue.get_nowait()
                st.session_state.latest_frame = fmsg.get("frame_b64", None)
            except queue.Empty:
                break

        # Drain the telemetry queue of any messages since the last rerun loop
        while not mqtt_queue.empty():
            try:
                sample = mqtt_queue.get_nowait()
                decision = evaluate(sample)
                flat = flatten(sample, decision)
                st.session_state.telemetry.append(flat)
                if flat["ocr"]:
                    st.session_state.latest_ocr = flat["ocr"]
                if flat["sign"] != "NONE":
                    st.session_state.sign_counts[flat["sign"]] = st.session_state.sign_counts.get(flat["sign"], 0) + 1
                push_alert(sample, decision)
                new_data = True
            except queue.Empty:
                break

        # No new messages this tick → reuse last known sample so the UI is stable
        if not new_data:
            if len(st.session_state.telemetry) > 0:
                flat = st.session_state.telemetry[-1]
                sample = {
                    "timestamp": flat["ts"],
                    "camera": {"road_sign": flat["sign"], "confidence": flat["confidence"],
                               "ocr_text": flat.get("ocr", "") or (str(flat["limit"]) if flat["limit"] else "")},
                    "ultrasonic": {"distance_cm": flat["distance"]},
                    "gps": {"speed_kmh": flat["speed"], "lat": flat["lat"], "lon": flat["lon"]},
                }
                # No co_driver field carried on `flat`, so re-derive it locally
                # with the same function the Pi uses. Keeps the banner stable
                # between live ticks.
                sample["co_driver"] = get_co_driver_advice(
                    flat["sign"], flat["speed"], flat["distance"], flat.get("limit")
                )
                decision = evaluate(sample)
            else:
                decision = {"status": "SAFE", "reasons": ["Awaiting live MQTT data…"],
                            "actuators": {"buzzer": False},
                            "voice": "Waiting for connection.", "speed_limit": None}

# Optional disk snapshot (rate-limited inside the helper)
if autosave:
    autosave_logs()


# ─────────────────────────────────────────────────────────────────────────────
# 9. HEADER + STATUS BANNER
# ─────────────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 2])
with hcol1:
    st.markdown("<p class='sdas-title'>SDAS — Smart Driver Assistance</p>", unsafe_allow_html=True)
    st.markdown("<p class='sdas-subtitle'>Real-time traffic-sign recognition · collision warning · speed monitoring</p>",
                unsafe_allow_html=True)
with hcol2:
    st.markdown(f"<p class='sdas-subtitle' style='text-align:right'>"
                f"📡 {source}  •  ⏱ {datetime.now(MYT).strftime('%H:%M:%S')} MYT  •  Pi 5 · 8GB"
                f"</p>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="status-banner status-{decision['status']}">
        <div>
            <div class="status-label">● {decision['status']}</div>
            <div class="status-reason">{' • '.join(decision['reasons'])}</div>
        </div>
        <div style="text-align:right;color:#fff;">
            <div style="font-size:0.8rem;opacity:0.85">CO-DRIVER</div>
            <div style="font-size:1.05rem;font-weight:600">🔊 {decision['voice']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 10. TABS
# ─────────────────────────────────────────────────────────────────────────────
ensure_state()

tabs = st.tabs([
    "ℹ️ About",
    "🏠 Overview",
    "📊 Sensors",
    "🛑 Sign Detection",
    "🚨 Alerts",
    "🗺️ GPS Map",
    "📈 Analytics",
    "⚙️ Connectivity",
])

df = pd.DataFrame(list(st.session_state.telemetry))
if not df.empty:
    df["ts"] = pd.to_datetime(df["ts"])


# ============== TAB 0: ABOUT ==============
with tabs[0]:
    st.markdown("""
    <div style='padding:4px 0 12px 0'>
        <div style='font-size:1.25rem;font-weight:700;color:var(--sdas-text)'>
            An affordable AI co-driver for Malaysian roads
        </div>
        <div style='color:var(--sdas-text-muted);font-size:0.95rem;max-width:780px;margin-top:6px'>
            SDAS runs entirely on a Raspberry Pi 5. A camera reads the road, an ultrasonic sensor
            watches the bumper, and GPS tracks speed and position. A rule-based decision engine
            fuses everything into a single SAFE / WARNING / CRITICAL state — and an AI co-driver
            speaks naturally to the driver instead of just flashing labels.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-h'>Hardware components</div>", unsafe_allow_html=True)
    for row_start in range(0, len(HARDWARE_SPEC), 3):
        row_items = HARDWARE_SPEC[row_start:row_start + 3]
        cols = st.columns(3)
        for j, hw in enumerate(row_items):
            with cols[j]:
                st.markdown(
                    f"""<div class='info-card' style='min-height:200px;display:flex;flex-direction:column'>
                        <div class='icon'>{hw['icon']}</div>
                        <h4>{hw['name']}</h4>
                        <p style='flex:1'>{hw['desc']}</p>
                    </div>
                    <div style='height:10px'></div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-h'>What the system can do</div>", unsafe_allow_html=True)
    ccols = st.columns(2)
    for i, (title, body) in enumerate(CAPABILITIES):
        with ccols[i % 2]:
            st.markdown(
                f"""<div class='info-card'>
                    <h4>{title}</h4>
                    <p>{body}</p>
                </div>
                <div style='height:10px'></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-h'>Road signs the system recognises</div>", unsafe_allow_html=True)
    by_risk = {"high": [], "medium": [], "low": []}
    for name, meta in SIGN_CATALOG.items():
        if name == "NONE":
            continue
        if meta["risk"] in by_risk:
            by_risk[meta["risk"]].append(name)

    # Row 1: High risk (left) + Low risk (right)
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        chips = "".join(f"<span class='sign-chip high'>{s}</span>" for s in by_risk["high"])
        st.markdown(
            f"""<div class='info-card'>
                <h4>🔴 High risk <span style='color:var(--sdas-text-muted);font-weight:500'>· {len(by_risk['high'])}</span></h4>
                <p style='margin-bottom:10px'>Mandatory action or immediate hazard</p>
                <div>{chips}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with rcol2:
        chips = "".join(f"<span class='sign-chip low'>{s}</span>" for s in by_risk["low"])
        st.markdown(
            f"""<div class='info-card'>
                <h4>🟢 Low risk <span style='color:var(--sdas-text-muted);font-weight:500'>· {len(by_risk['low'])}</span></h4>
                <p style='margin-bottom:10px'>Informational signs</p>
                <div>{chips}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Row 2: Medium risk (full width — the longest list)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    chips = "".join(f"<span class='sign-chip medium'>{s}</span>" for s in by_risk["medium"])
    st.markdown(
        f"""<div class='info-card'>
            <h4>🟡 Medium risk <span style='color:var(--sdas-text-muted);font-weight:500'>· {len(by_risk['medium'])}</span></h4>
            <p style='margin-bottom:10px'>Prohibition, caution, or geometry change</p>
            <div>{chips}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    total = sum(len(v) for v in by_risk.values())
    st.caption(f"Total recognisable classes: **{total}** (plus an idle 'no sign' state).")

    # ── Decision logic + Co-Driver voice script (full, for transparency) ──
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-h'>🧠 Decision logic — worst vote wins</div>",
                unsafe_allow_html=True)
    st.markdown(
        """<div class='info-card'>
            <p style='margin-bottom:10px'>Every tick, three independent input
            channels each vote on severity. The highest severity becomes the
            final dashboard state — once any channel says <b>CRITICAL</b>, the
            others cannot demote it. <b>SAFE</b> only survives if no channel
            fires at all.</p>
        </div>
        <div style='height:10px'></div>""",
        unsafe_allow_html=True,
    )
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        st.markdown(
            """<div class='info-card' style='min-height:290px'>
                <h4>1️⃣ Distance channel</h4>
                <p><i>Source: HC-SR04 ultrasonic</i></p>
                <div style='height:10px'></div>
                <p>
                <b>&lt; 20 cm</b> → <span style='color:#EF4444;font-weight:700'>CRITICAL</span><br/>
                <span style='color:#8B9BC0'>↳ matches buzzer tone A5 on the Pi</span>
                </p>
                <div style='height:10px'></div>
                <p>
                <b>&lt; 50 cm</b> → <span style='color:#F59E0B;font-weight:700'>WARNING</span><br/>
                <span style='color:#8B9BC0'>↳ matches buzzer tone A4 on the Pi</span>
                </p>
                <div style='height:10px'></div>
                <p><b>≥ 50 cm</b> → no contribution</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with lcol2:
        st.markdown(
            """<div class='info-card' style='min-height:290px'>
                <h4>2️⃣ Speed vs Limit channel</h4>
                <p><i>Source: GPS speed + last seen Speed-limit sign (OCR)</i></p>
                <div style='height:10px'></div>
                <p>
                <b>speed &gt; limit + 10</b> → <span style='color:#EF4444;font-weight:700'>CRITICAL</span>
                </p>
                <div style='height:10px'></div>
                <p>
                <b>speed &gt; limit</b> → <span style='color:#F59E0B;font-weight:700'>WARNING</span>
                </p>
                <div style='height:10px'></div>
                <p><b>speed ≤ limit</b> → no contribution</p>
                <div style='height:10px'></div>
                <p style='color:#8B9BC0;font-size:0.85rem'>Limit comes from the
                most recent OCR'd Speed-limit sign; the catalog provides an
                implied limit for some other classes (e.g. School zone → 30).</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with lcol3:
        st.markdown(
            """<div class='info-card' style='min-height:290px'>
                <h4>3️⃣ Sign-risk channel</h4>
                <p><i>Source: YOLO classification</i></p>
                <div style='height:10px'></div>
                <p>
                <b>high-risk sign in frame</b> → <span style='color:#F59E0B;font-weight:700'>WARNING</span><br/>
                <span style='color:#8B9BC0'>↳ e.g. Stop, No entry, Zebra crossing,
                Children, Train Gate, Obstruction…</span>
                </p>
                <div style='height:10px'></div>
                <p><b>medium / low / no sign</b> → no contribution</p>
                <div style='height:10px'></div>
                <p style='color:#8B9BC0;font-size:0.85rem'>Medium/low signs
                still produce a Co-Driver voice line, but do not push the
                dashboard state above SAFE on their own.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-h'>🗣️ Co-Driver voice script — priority order</div>",
                unsafe_allow_html=True)
    st.markdown(
        """<div class='info-card'>
            <p>A single function <code>get_co_driver_advice(sign, speed, distance, limit)</code>
            lives in both <code>pi_publisher.py</code> and <code>app.py</code> with
            identical logic. The Pi stamps the result onto every telemetry
            packet under the <code>co_driver</code> key; the simulator does the
            same. The dashboard never re-derives it.</p>
            <p>All <b>63 catalog states</b> (62 named signs + the idle NONE
            state) are mapped 1-to-1 — there are no silent fallbacks.</p>
        </div>
        <div style='height:10px'></div>""",
        unsafe_allow_html=True,
    )
    pcol1, pcol2 = st.columns(2)

    with pcol1:
        # ── ① Distance overrides ──────────────────────────────────────
        st.markdown(
            """<div class='info-card'>
                <h4>① Distance overrides &nbsp;<span style='color:#EF4444;font-size:0.8rem'>highest priority</span></h4>
                <p>Beat everything else — the sign and the speed don't matter
                if something is right in front of the bumper.</p>
                <div style='height:10px'></div>
                <p>
                <b>&lt; 20 cm</b><br/>
                <span style='color:#90E0EF'>→ "Collision risk detected. Brake immediately."</span>
                </p>
                <div style='height:10px'></div>
                <p>
                <b>&lt; 50 cm</b><br/>
                <span style='color:#90E0EF'>→ "Obstacle ahead. Reduce speed."</span>
                </p>
            </div>
            <div style='height:10px'></div>""",
            unsafe_allow_html=True,
        )

        # ── ③ Sign-specific lookup ────────────────────────────────────
        st.markdown(
            """<div class='info-card'>
                <h4>③ Sign-specific lookup &nbsp;<span style='color:#00B4D8;font-size:0.8rem'>62 classes</span></h4>
                <p>One-to-one mapping for every catalog class — no silent
                fallbacks. The full reference table is in the expander below.</p>
            </div>
            <div style='height:10px'></div>""",
            unsafe_allow_html=True,
        )

        # ── ④ Idle ────────────────────────────────────────────────────
        st.markdown(
            """<div class='info-card'>
                <h4>④ Idle &nbsp;<span style='color:#8B9BC0;font-size:0.8rem'>no sign in frame</span></h4>
                <p>
                <b>NONE</b> &nbsp;
                <span style='color:#90E0EF'>→ "Road clear."</span>
                </p>
                <p style='color:#8B9BC0;font-size:0.85rem'>An unknown class
                that isn't in the catalog falls through to
                <i>"Road conditions normal."</i> — by design, this should
                never happen with the trained model.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    with pcol2:
        # ── ② Speed-limit sign (tall card, spans the full right side) ─
        st.markdown(
            """<div class='info-card'>
                <h4>② Speed-limit sign &nbsp;<span style='color:#F59E0B;font-size:0.8rem'>dynamic</span></h4>
                <p>Uses the OCR-decoded limit when available; falls back to a
                60 km/h check if OCR misses the digits.</p>
                <div style='height:10px'></div>
                <p>
                <b>limit known &amp; speed &gt; limit</b><br/>
                <span style='color:#90E0EF'>→ "Speed limit {limit} detected. Current speed {N}. Reduce speed."</span>
                </p>
                <div style='height:10px'></div>
                <p>
                <b>limit known &amp; speed ≤ limit</b><br/>
                <span style='color:#90E0EF'>→ "Speed limit {limit} detected."</span>
                </p>
                <div style='height:10px'></div>
                <p>
                <b>limit unknown (OCR miss) &amp; speed &gt; 60</b><br/>
                <span style='color:#90E0EF'>→ "Speed limit sign detected. Current speed {N}. Reduce speed."</span>
                </p>
                <div style='height:10px'></div>
                <p>
                <b>limit unknown &amp; speed ≤ 60</b><br/>
                <span style='color:#90E0EF'>→ "Speed limit sign detected."</span>
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Full 62-sign voice-script reference table ─────────────────────
    VOICE_SCRIPT = {
        "Stop":                              "Stop sign ahead. Prepare to stop.",
        "No entry":                          "No entry ahead. Do not proceed.",
        "Children":                          "School zone ahead. Watch for children.",
        "Cow nearby":                        "Livestock on road. Slow down.",
        "Flagman ahead":                     "Flagman directing traffic. Obey signals.",
        "Level crossing with gates ahead":   "Railway crossing ahead. Prepare to stop.",
        "Train Gate":                        "Train gate ahead. Stop if closing.",
        "Traffic signals ahead":             "Traffic light ahead. Be ready to stop.",
        "Obstruction":                       "Obstruction on road. Reduce speed.",
        "Road work":                         "Road work ahead. Slow down.",
        "pedestrian crossing opt1":          "Pedestrian crossing. Watch for pedestrians.",
        "Zebra crossing":                    "Zebra crossing ahead. Give way to pedestrians.",
        "Bumps":                             "Speed bump. Slow down.",
        "Bumps ahead":                       "Bumps ahead. Reduce speed.",
        "Camera operation zone":             "Speed camera zone. Watch your speed.",
        "Chevron (left)":                    "Sharp turn left ahead.",
        "Chevron (right)":                   "Sharp turn right ahead.",
        "Construction sign":                 "Construction zone. Drive carefully.",
        "Crossroad":                         "Crossroad ahead. Watch for crossing traffic.",
        "Crossroad on the left":             "Crossroad on left ahead.",
        "Crossroad on the right":            "Crossroad on right ahead.",
        "Crosswind area":                    "Crosswind area. Grip the wheel firmly.",
        "Double Bend to Left Ahead":         "Double bend left ahead. Reduce speed.",
        "Double Bend to Right Ahead":        "Double bend right ahead. Reduce speed.",
        "Give way":                          "Give way to oncoming traffic.",
        "Height limit":                      "Height limit ahead. Check vehicle clearance.",
        "Horn Prohibited":                   "No horn zone.",
        "Left Bend Ahead":                   "Left bend ahead. Reduce speed.",
        "Narrow bridge":                     "Narrow bridge ahead. Drive carefully.",
        "No left turn":                      "Left turn prohibited.",
        "No right turn":                     "Right turn prohibited.",
        "No overtaking":                     "Overtaking prohibited.",
        "No parking":                        "No parking allowed here.",
        "No Stopping":                       "No stopping allowed here.",
        "No U-turns":                        "U-turn prohibited.",
        "Other dangers nearby":              "Hazard ahead. Stay alert.",
        "Pass either side":                  "Pass either side of obstacle.",
        "Pass on the left":                  "Keep left to pass.",
        "Pass on the right":                 "Keep right to pass.",
        "Right Bend Ahead":                  "Right bend ahead. Reduce speed.",
        "Road cones":                        "Road cones ahead. Reduce speed.",
        "Road narrows on the left":          "Road narrows on left.",
        "Road narrows on the right":         "Road narrows on right.",
        "Roadway diverges":                  "Road splits ahead. Choose lane early.",
        "Roundabout ahead":                  "Roundabout ahead. Give way.",
        "Slippery road":                     "Slippery road. Reduce speed.",
        "Speed limit":                       "(dynamic — see ② above)",
        "T-junction":                        "T-junction ahead. Prepare to turn.",
        "Traffic from Left Merges Ahead":    "Traffic merging from left ahead.",
        "Traffic from Right Merges Ahead":   "Traffic merging from right ahead.",
        "Traffic merging from the left":     "Traffic merging from left.",
        "Traffic merging from the right":    "Traffic merging from right.",
        "Traffic merging to the left":       "Traffic merging to left.",
        "U turn":                            "U-turn permitted ahead.",
        "Weight limit":                      "Weight limit ahead. Check vehicle weight.",
        "Bicycle lane":                      "Bicycle lane. Share the road.",
        "Bus stop":                          "Bus stop ahead.",
        "Expressway signs 1":                "Expressway ahead.",
        "Expressway signs 2":                "Expressway information ahead.",
        "Motorcycles only":                  "Motorcycles only lane.",
        "Parking area":                      "Parking area ahead.",
        "Towing area":                       "Towing zone.",
    }

    with st.expander("📋 Full voice-script reference — all 62 sign classes", expanded=False):
        # Build table rows grouped by risk
        risk_label = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        rows_html = ""
        for sign_name, voice_line in VOICE_SCRIPT.items():
            cat_meta = SIGN_CATALOG.get(sign_name, SIGN_CATALOG.get(sign_name, {"risk": "?"}))
            risk = cat_meta.get("risk", "?")
            dot = risk_label.get(risk, "⚪")
            rows_html += (
                f"<tr>"
                f"<td style='padding:5px 10px;border-bottom:1px solid rgba(0,180,216,0.10);color:#E2EAFC;white-space:nowrap'>{dot} {sign_name}</td>"
                f"<td style='padding:5px 10px;border-bottom:1px solid rgba(0,180,216,0.10);color:#90E0EF;font-style:italic'>\"{voice_line}\"</td>"
                f"</tr>"
            )
        st.markdown(
            f"""<div style='max-height:480px;overflow-y:auto;border:1px solid rgba(0,180,216,0.15);border-radius:8px;background:#1C2541'>
                <table style='width:100%;border-collapse:collapse;font-size:0.88rem'>
                    <thead>
                        <tr style='background:rgba(0,180,216,0.08)'>
                            <th style='padding:8px 10px;text-align:left;color:#C9D8F0;font-weight:600;position:sticky;top:0;background:#1C2541'>Sign class</th>
                            <th style='padding:8px 10px;text-align:left;color:#C9D8F0;font-weight:600;position:sticky;top:0;background:#1C2541'>Co-Driver says…</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption(f"{len(VOICE_SCRIPT)} sign classes mapped · Speed limit is dynamic (see ② above) · "
                   "NONE idle state → \"Road clear.\"  · Unknown sign → \"Road conditions normal.\"")


# ============== TAB 1: OVERVIEW ==============
with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    spd = flat["speed"] if flat else 0
    dist = flat["distance"] if flat else 0
    lim = flat.get("limit") if flat else "—"
    sign_now = flat["sign"] if flat else "—"

    c1.metric("Speed", f"{spd:.1f} km/h",
              delta=f"limit {lim}" if lim else None,
              delta_color="inverse" if isinstance(lim, (int, float)) and spd > lim else "normal")
    c2.metric("Distance", f"{dist:.0f} cm",
              delta=f"warn <{CONFIG['DISTANCE_WARN_CM']}",
              delta_color="inverse" if dist < CONFIG["DISTANCE_WARN_CM"] else "normal")

    # Custom "Detected sign" card — st.metric truncates long values like
    # "Level crossing with gates ahead" with an ellipsis. This card wraps and
    # auto-shrinks the font for longer names so the full sign is always
    # legible.
    display_sign = sign_now if sign_now and sign_now != "NONE" else "—"
    name_len     = len(display_sign)
    if name_len <= 12:
        sign_font_size = "1.55rem"
    elif name_len <= 20:
        sign_font_size = "1.20rem"
    elif name_len <= 28:
        sign_font_size = "1.00rem"
    else:
        sign_font_size = "0.88rem"
    c3.markdown(
        f"""
        <div style='
            background:#1C2541;
            border:1px solid rgba(0,180,216,0.18);
            border-radius:10px;
            padding:14px 18px;
            min-height:110px;
            display:flex;
            flex-direction:column;
            justify-content:space-between;
        '>
            <div style='color:#8B9BC0;font-size:0.85rem;font-weight:400;line-height:1.2'>
                Detected sign
            </div>
            <div style='
                color:#E2EAFC;
                font-size:{sign_font_size};
                font-weight:700;
                line-height:1.2;
                margin-top:6px;
                word-break:break-word;
                white-space:normal;
            '>
                {display_sign}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c4.metric("Risk level", decision["status"])

    st.markdown("<div class='section-h'>Actuator state</div>", unsafe_allow_html=True)
    a = decision["actuators"]
    voice_line = decision.get("voice", "—")
    st.markdown(
        f"<div style='font-size:0.95rem'>"
        f"<span class='pill pill-{'on' if a['buzzer'] else 'off'}'>🔔 Buzzer {'ON' if a['buzzer'] else 'OFF'}</span>"
        f"<span class='pill pill-on'>🔊 Co-Driver “{voice_line}”</span>"
        f"</div>", unsafe_allow_html=True
    )

    st.markdown("<div class='section-h'>Live trend (last 60 samples)</div>", unsafe_allow_html=True)
    if not df.empty:
        recent = df.tail(60)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["ts"], y=recent["speed"], name="Speed (km/h)",
                                 line=dict(color="#90E0EF", width=2)))
        fig.add_trace(go.Scatter(x=recent["ts"], y=recent["distance"], name="Distance (cm)",
                                 line=dict(color="#00B4D8", width=2), yaxis="y2"))
        fig = style_fig(fig, height=280)
        fig.update_layout(
            yaxis=dict(title="Speed", side="left"),
            yaxis2=dict(title="Distance", overlaying="y", side="right",
                        gridcolor="rgba(128,128,128,0.0)"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Awaiting first samples…")


# ============== TAB 2: SENSORS ==============
with tabs[2]:
    st.markdown("<div class='section-h'>Raw telemetry — rolling window</div>", unsafe_allow_html=True)
    if not df.empty:
        display_cols = ["ts", "status", "sign", "confidence", "ocr", "limit", "speed", "distance", "lat", "lon", "co_driver"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].tail(40).iloc[::-1], use_container_width=True, height=420)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Latest JSON payload**")
            if sample:
                st.code(json.dumps(sample, indent=2), language="json")
        with col_b:
            st.markdown("**Sensor health**")
            st.write("📷 Camera (YOLO + EasyOCR): ✅ streaming")
            st.write("📏 HC-SR04 Ultrasonic: ✅ streaming")
            st.write("🛰️ NEO-6M GPS: ✅ streaming")
            st.write(f"🔄 Sample rate: ~{1/refresh:.1f} Hz")
    else:
        st.info("No data yet.")


# ============== TAB 3: SIGN DETECTION ==============
with tabs[3]:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        # ── Live camera feed (MQTT frame) ──────────────────────────────────
        st.markdown("<div class='section-h'>📷 Live Camera Feed</div>", unsafe_allow_html=True)
        frame_b64 = st.session_state.get("latest_frame")
        if frame_b64:
            try:
                frame_bytes = base64.b64decode(frame_b64)
                st.image(frame_bytes, caption="Annotated feed — YOLO bounding boxes + OCR overlay",
                         use_container_width=True)
            except Exception:
                st.warning("Frame decode error. Waiting for next frame…")
        elif source == "Simulation":
            st.info("Live camera feed is only available in MQTT (live) mode.")
        else:
            st.info("No camera frame yet. Make sure the Pi publisher is running and "
                    "publishing to `sdas/pi5/01/frame`.")

    with col_r:
        # ── Current detection card ────────────────────────────────────────
        st.markdown("<div class='section-h'>🛑 Current Detection</div>", unsafe_allow_html=True)
        if sample and sample["camera"]["road_sign"] != "NONE":
            cam  = sample["camera"]
            sign = cam["road_sign"]
            meta = SIGN_CATALOG.get(sign, {"risk": "none", "action": "Unknown sign."})

            # OCR is only meaningful for the "Speed limit" sign — gate the
            # display so a stale 50 km/h from a previous frame doesn't get
            # rendered next to a Stop sign or anything else.
            ocr_val = ""
            if sign == "Speed limit":
                ocr_val = cam.get("ocr_text", "") or st.session_state.get("latest_ocr", "")

            risk_colors = {"high": "#EF4444", "medium": "#F59E0B", "low": "#00B4D8", "none": "#8B9BC0"}
            risk_color  = risk_colors.get(meta["risk"], "#8B9BC0")
            ocr_html = (
                f"<div style='color:#00B4D8;font-size:1.1rem;font-weight:700;margin:4px 0'>"
                f"🔢 OCR: {ocr_val}</div>"
                if (sign == "Speed limit" and ocr_val) else ""
            )
            st.markdown(
                f"""
                <div class='info-card'>
                    <div style='font-size:1.9rem;font-weight:700;color:#E2EAFC'>{sign}</div>
                    <div style='margin:6px 0;'>
                        <span style='color:{risk_color};font-weight:600;text-transform:uppercase;font-size:0.85rem'>
                            {meta['risk'].upper()} RISK
                        </span>
                        &nbsp;·&nbsp;
                        <span style='color:#8B9BC0;font-size:0.85rem'>conf {cam['confidence']:.2f}</span>
                    </div>
                    {ocr_html}
                    <div class='codriver-box'><b>Recommended action:</b> {meta['action']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No sign in current frame.")


# ============== TAB 4: ALERTS ==============
with tabs[4]:
    st.markdown("<div class='section-h'>Event log</div>", unsafe_allow_html=True)
    alerts_buf = st.session_state.get("alerts", deque())
    if alerts_buf:
        adf = pd.DataFrame(list(alerts_buf)).iloc[::-1]
        # Friendly column order with OCR right next to the sign
        col_order = ["timestamp", "severity", "reason", "sign", "ocr", "limit", "speed", "distance", "co_driver"]
        adf = adf[[c for c in col_order if c in adf.columns]]

        def color_sev(v):
            if v == "CRITICAL":
                return "background-color: #DC2626; color: #FFFFFF; font-weight: 700"
            if v == "WARNING":
                return "background-color: #D97706; color: #FFFFFF; font-weight: 700"
            return ""

        st.dataframe(
            adf.style.map(color_sev, subset=["severity"]),
            use_container_width=True, height=460,
        )
        st.caption(f"Total events: {len(adf)} · Critical: "
                   f"{(adf['severity']=='CRITICAL').sum()} · Warning: "
                   f"{(adf['severity']=='WARNING').sum()}")
    else:
        st.success("No alerts — all systems nominal.")


# ============== TAB 5: GPS MAP ==============
with tabs[5]:
    st.markdown("<div class='section-h'>Vehicle trajectory</div>", unsafe_allow_html=True)
    if not df.empty and len(df) > 1:
        path = df[["lon", "lat"]].values.tolist()
        layer_path = pdk.Layer(
            "PathLayer",
            data=[{"path": path, "color": [79, 141, 253]}],
            get_path="path", get_color="color", width_scale=4, width_min_pixels=3,
        )
        layer_current = pdk.Layer(
            "ScatterplotLayer",
            data=[{"position": [df.iloc[-1]["lon"], df.iloc[-1]["lat"]]}],
            get_position="position", get_radius=30,
            get_fill_color=[239, 68, 68] if decision["status"] == "CRITICAL"
                           else [245, 158, 11] if decision["status"] == "WARNING"
                           else [0, 180, 216],
            pickable=True,
        )
        view = pdk.ViewState(
            latitude=df.iloc[-1]["lat"], longitude=df.iloc[-1]["lon"],
            zoom=15, pitch=40,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer_path, layer_current],
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/dark-v10",
        ))
        m1, m2, m3 = st.columns(3)
        m1.metric("Latitude", f"{df.iloc[-1]['lat']:.5f}")
        m2.metric("Longitude", f"{df.iloc[-1]['lon']:.5f}")
        m3.metric("Track points", len(df))
    else:
        st.info("Collecting GPS fixes…")


# ============== TAB 6: ANALYTICS ==============
with tabs[6]:
    if df.empty:
        st.info("No data to analyse yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-h'>Speed over time</div>", unsafe_allow_html=True)
            fig = px.line(df, x="ts", y="speed")
            fig.update_traces(line_color="#90E0EF")
            st.plotly_chart(style_fig(fig, height=300), use_container_width=True)
        with col2:
            st.markdown("<div class='section-h'>Distance over time</div>", unsafe_allow_html=True)
            fig = px.line(df, x="ts", y="distance")
            fig.update_traces(line_color="#00B4D8")
            st.plotly_chart(style_fig(fig, height=300), use_container_width=True)

        st.markdown("<div class='section-h'>Risk surface — speed vs distance</div>", unsafe_allow_html=True)
        fig = px.scatter(
            df, x="distance", y="speed", color="status",
            color_discrete_map={"SAFE": "#00B4D8", "WARNING": "#F59E0B", "CRITICAL": "#EF4444"},
            hover_data=["sign", "ts"],
        )
        fig.add_vrect(x0=0, x1=CONFIG["DISTANCE_CRIT_CM"], fillcolor="#EF4444",
                      opacity=0.12, line_width=0, annotation_text="critical zone",
                      annotation_position="top left")
        fig.add_vrect(x0=CONFIG["DISTANCE_CRIT_CM"], x1=CONFIG["DISTANCE_WARN_CM"],
                      fillcolor="#F59E0B", opacity=0.10, line_width=0,
                      annotation_text="warning zone", annotation_position="top left")
        st.plotly_chart(style_fig(fig, height=400, margin=(10, 10, 30, 10)),
                        use_container_width=True)

        # ── Detection frequency (moved here from Sign Detection tab) ──────
        st.markdown("<div class='section-h'>Detection frequency this session</div>",
                    unsafe_allow_html=True)
        sign_counts = st.session_state.get("sign_counts", {})
        if sign_counts:
            sc = pd.DataFrame(
                [{"sign": k, "count": v} for k, v in sign_counts.items()]
            ).sort_values("count", ascending=True)
            fig = px.bar(sc, x="count", y="sign", orientation="h",
                         color="count", color_continuous_scale="Teal")
            fig = style_fig(fig, height=max(380, len(sc) * 26))
            fig.update_layout(coloraxis_showscale=False,
                              yaxis_title=None, xaxis_title="Detections")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No signs detected yet.")


# ============== TAB 7: CONNECTIVITY ==============
with tabs[7]:
    st.markdown("<div class='section-h'>MQTT topic plan</div>", unsafe_allow_html=True)
    st.code(MQTT_TOPIC_PLAN, language="text")

    st.markdown("<div class='section-h'>Pi 5 publisher</div>", unsafe_allow_html=True)
    st.markdown("""
    The full publisher already lives in **`pi_publisher.py`** in this project.
    It runs on the Pi 5 and streams to the same broker the dashboard subscribes to,
    so switching the sidebar to **MQTT (live)** is all you need on this side.

    **On the Pi:**
    ```bash
    sudo pigpiod                  # accurate ultrasonic timing (recommended)
    python pi_publisher.py        # YOLO + EasyOCR + HC-SR04 + NEO-6M @ ~1 Hz
    ```

    **What it does each tick:**
    - Captures one camera frame, runs the custom YOLO model at `conf ≥ 0.55`
      and EasyOCR on speed-limit crops.
    - Reads the HC-SR04 (and drives the buzzer locally: A5 < 20 cm, A4 < 50 cm).
    - Drains the NEO-6M serial buffer for the latest GPRMC / GNRMC fix.
    - Publishes telemetry JSON to `sdas/pi5/01/telemetry` (QoS 1) and the
      annotated JPEG (base64, quality 50) to `sdas/pi5/01/frame` (QoS 0).
    """)

    st.markdown("<div class='section-h'>Telemetry payload</div>", unsafe_allow_html=True)
    st.code(json.dumps({
        "timestamp":  "2026-06-19T16:30:12+08:00",
        "camera":     {"road_sign": "Speed limit", "confidence": 0.94, "ocr_text": "60km/h"},
        "ultrasonic": {"distance_cm": 78.4},
        "gps":        {"speed_kmh": 72.0, "lat": 3.13901, "lon": 101.68690},
        "co_driver":  "Speed limit 60 detected. Current speed 72. Reduce speed.",
    }, indent=2), language="json")
    st.caption("Note: the publisher emits OCR like `\"60km/h\"`; the dashboard "
               "strips non-digits, so bare `\"60\"` from the simulator also works. "
               "The `co_driver` line is generated by `get_co_driver_advice()` — "
               "the same function lives in `app.py` so simulation mode produces "
               "identical voice lines.")

    st.markdown("<div class='section-h'>Frame payload</div>", unsafe_allow_html=True)
    st.code(json.dumps({
        "timestamp": "2026-06-19T16:30:12+08:00",
        "frame_b64": "<base64 JPEG, quality 50>",
    }, indent=2), language="json")

    st.markdown("<div class='section-h'>Pre-flight checks</div>", unsafe_allow_html=True)
    st.markdown("""
    - **Sign labels** must match `SIGN_CATALOG` keys exactly. Print
      `model.names` once on the Pi and diff against the catalog —
      `"Speed limit"`, not `"speed_limit"` or `"SpeedLimit"`.
    - **EasyOCR** uses `gpu=False` on Pi 5 (no CUDA).
    - **Broker** is the public `broker.hivemq.com:1883` — fine for a demo,
      not for production. Use an authenticated broker + TLS for real use.
    - **Buzzer logic** lives on the Pi (`pi_publisher.py`). Dashboard
      threshold sliders do not change the physical alarm.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 11. AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(refresh)
    st.rerun()
