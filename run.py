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
        f.write("GM102B_slope,GM302B_slope,GM502B_slope,GM702B_slope,"
                "GM102B_mean,GM302B_mean,GM502B_mean,GM702B_mean,"
                "GM102B_CV,GM302B_CV,GM502B_CV,GM702B_CV,color\n")

def read_serial():
    """Reads serial data, processes it into slopes, means, CVs, and writes to a CSV file."""
    entry_count = 0  # Count the number of recorded entries

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Allow time for connection
        print("Connected to Serial Port:", SERIAL_PORT)
    except serial.SerialException as e:
        print(f"Failed to connect: {e}")
        stop_flag.set()  # Stop program if serial fails
        return

    while entry_count < MAX_ENTRIES and not stop_flag.is_set():  # Stop after 10 entries
        try:
            readings = { "GM102B": [], "GM302B": [], "GM502B": [], "GM702B": [] }

            # Collect 15 valid readings
            valid_count = 0
            while valid_count < 15 and not stop_flag.is_set():
                line = ser.readline().decode(errors='replace').strip()
                values = line.split(',')

                if len(values) == 4 and all(v.replace('.', '', 1).isdigit() for v in values):
                    values = list(map(float, values))  # Convert to float
                    readings["GM102B"].append(values[0])
                    readings["GM302B"].append(values[1])
                    readings["GM502B"].append(values[2])
                    readings["GM702B"].append(values[3])
                    valid_count += 1  # Count only valid readings

                time.sleep(1)  # Small delay for stable readings

            # Compute medians every 3 values
            slopes, means, cvs = [], [], []
            for key in readings.keys():
                medians = [np.median(readings[key][i:i+3]) for i in range(0, 15, 3)]
                slope = np.polyfit(range(1, 6), medians, 1)[0] if len(medians) == 5 else 0
                mean_val = np.mean(readings[key])  # Calculate mean
                cv_val = np.std(readings[key]) / mean_val if mean_val != 0 else 0  # Calculate CV

                slopes.append(slope)
                means.append(mean_val)
                cvs.append(cv_val)

            print(f"Saved entry {entry_count + 1}/{MAX_ENTRIES}: Slopes, Means, and CVs")

            # Save to buffer for PCA
            data_buffer.append(slopes + means + cvs)

            # Append to CSV file with color
            with open(CSV_FILE, 'a') as f:
                f.write(",".join(map(str, slopes + means + cvs)) + f",{point_color}\n")

            entry_count += 1  # Increment entry count

        except Exception as e:
            print(f"Serial Read Error: {e}")
            break

    # **Disconnect serial and exit the program properly**
    print("Reached 10 entries, disconnecting serial and stopping program...")
    ser.close()
    stop_flag.set()  # Signal the PCA thread to stop
    os._exit(0)  # Forcefully terminate the entire program

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
                transformed_data = pca.fit_transform(df.iloc[:, :-1])  # Exclude color column
                
                # Plot points with color from CSV
                ax.clear()
                ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=df['color'], alpha=0.5)
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
