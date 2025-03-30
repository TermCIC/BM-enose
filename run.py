import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from collections import deque
import threading
import time
import sys  # Needed for safe program termination

# Serial port configuration
SERIAL_PORT = 'COM8'  # Change to match your device
BAUD_RATE = 9600
CSV_FILE = 'enose_slope_readings.csv'
MAX_ENTRIES = 10  # Stop after collecting 10 entries

# Storage for real-time PCA
buffer_size = 50  # Keep last 50 readings
data_buffer = deque(maxlen=buffer_size)

# Global flag to stop threads
stop_flag = threading.Event()

# Ask user for plot color
print("Enter the color for PCA plot points (e.g., red, blue, green, etc.): ")
point_color = input().strip()

# Check if the CSV file exists, create it if not
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w') as f:
        f.write("GM102B_rel,GM302B_rel,GM502B_rel,GM702B_rel,color\n")


def send_command(ser, command):
    """Sends a command to the serial device."""
    try:
        if ser and ser.is_open:
            ser.write((command + '\n').encode())  # Ensure newline for command
            print(f">>> Sent command: {command}")
        else:
            print("Serial port not open.")
    except Exception as e:
        print(f"Error sending command: {e}")


def read_serial():
    """Controls pumps, collects raw e-nose data, and saves baseline-subtracted values to CSV."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("Connected to Serial Port:", SERIAL_PORT)

        # Initial state: All pumps OFF
        send_command(ser, "PUMP1 OFF")
        send_command(ser, "PUMP2 OFF")

        print("Step 1: Running control air (PUMP2 ON) for 60 seconds...")
        send_command(ser, "PUMP2 ON")
        time.sleep(60)
        send_command(ser, "PUMP2 OFF")

        print("Step 2: Running sample air (PUMP1 ON) for 60 seconds and collecting data...")
        send_command(ser, "PUMP1 ON")

        raw_readings = []  # To store all sensor readings (list of 4-element lists)
        start_time = time.time()

        while time.time() - start_time < 60 and not stop_flag.is_set():
            line = ser.readline().decode(errors='replace').strip()
            values = line.split(',')

            if len(values) == 4 and all(v.replace('.', '', 1).isdigit() for v in values):
                values = list(map(float, values))
                raw_readings.append(values)
                print(values)
            time.sleep(1)

        send_command(ser, "PUMP1 OFF")

        print("Step 3: Reset signal (PUMP2 ON) for 60 seconds...")
        send_command(ser, "PUMP2 ON")
        time.sleep(60)
        send_command(ser, "PUMP2 OFF")

        # === Baseline-subtracted processing ===
        if not raw_readings:
            print("No valid readings received.")
        else:
            v0 = raw_readings[0]  # First sample = baseline
            print(f"Baseline (V0): {v0}")

            with open(CSV_FILE, 'a') as f:
                for reading in raw_readings:
                    diff = [v - b for v, b in zip(reading, v0)]
                    f.write(",".join(map(str, diff)) + f",{point_color}\n")

    except Exception as e:
        print(f"Serial Read Error: {e}")
    finally:
        print("Cleaning up and closing serial port.")
        ser.close()
        stop_flag.set()
        os._exit(0)


def plot_pca():
    """Runs in the main thread to update PCA plot."""
    plt.ion()
    fig, ax = plt.subplots()
    pca = PCA(n_components=2)

    while not stop_flag.is_set():
        try:
            # Read CSV and check for missing values
            df = pd.read_csv(CSV_FILE)

            # Drop any rows with NaN values
            df.dropna(inplace=True)

            if len(df) > 2:
                transformed_data = pca.fit_transform(
                    df.iloc[:, :-1])  # Exclude color column

                # Plot points with color from CSV
                ax.clear()
                ax.scatter(
                    transformed_data[:, 0], transformed_data[:, 1], c=df['color'], alpha=0.5)
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('Real-Time PCA of Gas Sensor Slopes & Means')
                plt.draw()
                plt.pause(1)
        except Exception as e:
            print(f"PCA Plot Error: {e}")

    print("PCA thread stopped.")


if __name__ == "__main__":
    # Start serial reading in a background thread
    serial_thread = threading.Thread(target=read_serial, daemon=True)
    serial_thread.start()

    # Run the plotting in the main thread
    plot_pca()
