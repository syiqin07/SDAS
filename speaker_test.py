from gpiozero import DistanceSensor
import os
import time

sensor = DistanceSensor(
    echo=23,
    trigger=18,
    max_distance=4
)

last_state = ""

while True:
    try:
        distance_cm = sensor.distance * 100

        print(f"Distance: {distance_cm:.1f} cm")

        if distance_cm < 30:
            if last_state != "warning":
                os.system('espeak-ng "Warning. Obstacle ahead."')
                last_state = "warning"

        else:
            last_state = "safe"

        time.sleep(1)

    except Exception as e:
        print("Error:", e)
        time.sleep(1)