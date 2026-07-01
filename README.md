# Smart Driver Assistance System (SDAS)

## Overview

The Smart Driver Assistance System (SDAS) is an AI-powered driving assistant developed using a Raspberry Pi 5. The system integrates Computer Vision, GPS, Ultrasonic sensing, MQTT communication, and a Streamlit dashboard to provide real-time driving assistance.

The system is capable of:

* Detecting traffic signs using YOLO
* Reading text from road signs using EasyOCR
* Measuring obstacle distance using an HC-SR04 Ultrasonic Sensor
* Obtaining live GPS location and vehicle speed
* Displaying all sensor data on a real-time Streamlit dashboard
* Providing voice alerts through a Bluetooth speaker

---

# Hardware Requirements

* Raspberry Pi 5 (64-bit Raspberry Pi OS)
* Raspberry Pi Camera / USB Webcam
* Neo-6M GPS Module
* HC-SR04 Ultrasonic Sensor
* Bluetooth Speaker
* Internet Connection (WiFi or Mobile Hotspot)
* Laptop with VS Code (recommended)

---

# Software Requirements

* Raspberry Pi OS (64-bit)
* Python 3.13
* VS Code
* Remote SSH Extension
* Streamlit
* MQTT
* OpenCV
* Ultralytics YOLO
* EasyOCR
* gpiozero
* pyserial
* pynmea2
* paho-mqtt

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Hardware Connections

## GPS Module (Neo-6M)

| GPS Module | Raspberry Pi |
| ---------- | ------------ |
| VCC        | 5V           |
| GND        | GND          |
| TX         | GPIO15 (RXD) |
| RX         | GPIO14 (TXD) |

---

## HC-SR04 Ultrasonic Sensor

| Ultrasonic | Raspberry Pi |
| ---------- | ------------ |
| VCC        | 5V           |
| GND        | GND          |
| TRIG       | GPIO23       |
| ECHO       | GPIO24       |

---

## Camera

Connect the USB camera to any USB port on the Raspberry Pi.

---

## Bluetooth Speaker

Pair the Bluetooth speaker using:

```bash
bluetoothctl
```

Commands:

```text
power on
agent on
default-agent
scan on
pair <MAC_ADDRESS>
trust <MAC_ADDRESS>
connect <MAC_ADDRESS>
```

---

# Network Setup

## Recommended Setup

```
             Mobile Hotspot / WiFi
                     │
      ┌──────────────┼───────────────┐
      │              │               │
 Raspberry Pi     Developer PC     Viewer PC
      │              │               │
      │          VS Code SSH      Dashboard
```

All devices must be connected to the same network.

---

# Connecting to Raspberry Pi

SSH from VS Code or Terminal.

Example:

```bash
ssh wcc@172.20.10.2
```

or configure

```
Host raspberrypi
HostName 172.20.10.2
User wcc
```

Then simply connect using

```
ssh raspberrypi
```

---

# Starting the Project

Navigate to the project directory.

```bash
cd ~/SDAS
```

Activate the virtual environment.

```bash
source venv/bin/activate
```

Start the SDAS system.

```bash
python pi_publisher2.py
```

**Only this command is required.**

The following services will automatically start:

* Camera
* YOLO Detection
* EasyOCR
* GPS
* Ultrasonic Sensor
* MQTT Publisher
* Streamlit Dashboard

---

# Opening the Dashboard

After `pi_publisher2.py` starts successfully, Streamlit will display a URL similar to:

```
http://172.20.10.2:8501
```

Open this URL in any web browser.
Dashboard Streamlit URL: https://sdas-malaysia-roadsign-detection.streamlit.app/

Any device connected to the same WiFi or hotspot can access the dashboard.
or RUN: 
```
streamlit run app.py
```

---

# GPS Notes

The GPS module requires a satellite fix.

For the first startup:

* Place the GPS antenna facing upward.
* Test outdoors or near an open window.
* The first cold start may take several minutes.

When connected successfully, GPS outputs:

* Latitude
* Longitude
* Speed (km/h)

---

# Camera Notes

The project uses:

* YOLO for traffic sign detection
* EasyOCR for reading road signs

The camera runs headless (no monitor required).

---

# Dashboard Features

The dashboard displays:

* Live Camera Detection
* Detected Traffic Sign
* OCR Text
* GPS Latitude
* GPS Longitude
* Vehicle Speed
* Obstacle Distance
* System Status
* MQTT Connection Status

---

# MQTT Broker

Broker:

```
broker.hivemq.com
```

Communication protocol:

```
MQTT
```

---

# Folder Structure

```
SDAS/
│
├── app.py
├── pi_publisher2.py
├── best.pt
├── requirements.txt
├── README.md
│
├── models/
│
├── images/
│
├── sounds/
│
├── venv/
│
└── ...
```

---

# Troubleshooting

## Camera not detected

Check:

```bash
ls /dev/video*
```

---

## GPS not working

Verify UART:

```bash
sudo raspi-config
```

Interface Options

Serial Port

Login Shell

```
No
```

Serial Hardware

```
Yes
```

Test GPS:

```bash
cat /dev/serial0
```

You should receive NMEA sentences.

---

## Ultrasonic Sensor

If no echo is detected:

* Check wiring
* Verify Trigger and Echo pins
* Ensure object is within sensing distance

---

## Bluetooth Speaker

Reconnect using:

```bash
bluetoothctl
```

---

## Dashboard inaccessible

Ensure:

* Raspberry Pi and client device are connected to the same network.
* Streamlit is running.
* Port 8501 is accessible.

---

# Shutting Down

Exit the application using:

```
Ctrl + C
```

Safely power off the Raspberry Pi:

```bash
sudo shutdown now
```

Wait until the green LED stops blinking before disconnecting the power supply.

---

# Authors

Faculty of Artificial Intelligence

Universiti Teknologi Malaysia

Bachelor of Artificial Intelligence

Final Year Project

Smart Driver Assistance System (SDAS)
