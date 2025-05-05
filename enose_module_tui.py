from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Input, Button, Log
from textual.reactive import reactive
from textual_plot import HiResMode, PlotWidget
import threading
import time
import numpy as np
import serial
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import csv
from settings import find_working_port
import os
import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    BASE_PATH = Path(sys._MEIPASS)
else:
    BASE_PATH = Path(__file__).parent

SETTINGS_PATH = BASE_PATH / "settings.py"
CSS_PATH = BASE_PATH / "enose_app.css"
CONTROL_CSV_PATH = BASE_PATH / "enose_control_summary.csv"

# Serial port configuration
SERIAL_PORT = find_working_port()  # Change to match your device
BAUD_RATE = 9600
READING_CSV_FILE = 'enose_readings.csv'
CONTROL_CSV_FILE = 'enose_control_summary.csv'
CORRECTION_CSV_FILE = 'enose_correction_coefficients.csv'

# Global serial connection object
ser = None


class ENoseApp(App):
    CSS_PATH = "enose_app.css"
    treatment_name = reactive("")
    run_count = reactive(1)
    reading_thread = None
    stop_flag = threading.Event()
    pause_flag = threading.Event()
    x_data = []
    y_data = []

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Container(id="left_panel"):
                yield Input(placeholder="Enter treatment name", id="treatment_input")
                with Horizontal(id="run_input_row"):
                    yield Input(placeholder="e.g. 3", id="run_input")
                    yield Button("+", id="inc_button")
                    yield Button("-", id="dec_button")
                with Horizontal(id="buttons_row"):
                    yield Button("Start", id="run_button")
                    yield Button("Cancel", id="cancel_button")
                    yield Button("Exit", id="exit_button")
                with Container(id="plot_box"):
                    yield PlotWidget(id="plot")
            with Container(id="treatment_panel"):
                yield Static("Treatments will appear here...", id="treatment_list")
            with Container(id="right_panel"):
                yield Log(highlight=True, id="reading_log")

    def init_csv_files(self):
        log = self.query_one("#reading_log", Log)
        # Check and create CONTROL_CSV_FILE
        if not os.path.exists(CONTROL_CSV_FILE):
            with open(CONTROL_CSV_FILE, 'w') as f:
                f.write(
                    "GM102B_mean,GM302B_mean,GM502B_mean,GM702B_mean,"
                    "temperature_mean,humidity_mean,"
                    "GM102B_max,GM302B_max,GM502B_max,GM702B_max,temperature_max,humidity_max,"
                    "GM102B_min,GM302B_min,GM502B_min,GM702B_min,temperature_min,humidity_min,"
                    "GM102B_sd,GM302B_sd,GM502B_sd,GM702B_sd,temperature_sd,humidity_sd,"
                    "GM102B_final,GM302B_final,GM502B_final,GM702B_final,temperature_final,humidity_final,"
                    "treatment,timestamp\n"
                )
            log.write_line(f"✅ Created: {os.path.abspath(CONTROL_CSV_FILE)}")
        else:
            log.write_line(f"📄 Exists: {os.path.abspath(CONTROL_CSV_FILE)}")

        # Check and create CORRECTION_CSV_FILE
        if not os.path.exists(CORRECTION_CSV_FILE):
            with open(CORRECTION_CSV_FILE, 'w') as f:
                f.write("sensor,a,b,intercept,treatment,timestamp\n")
            log.write_line(f"✅ Created: {os.path.abspath(CORRECTION_CSV_FILE)}")
        else:
            log.write_line(f"📄 Exists: {os.path.abspath(CORRECTION_CSV_FILE)}")

        # Check and create READING_CSV_FILE
        if not os.path.exists(READING_CSV_FILE):
            with open(READING_CSV_FILE, 'w') as f:
                f.write(
                    "GM102B_rel,GM302B_rel,GM502B_rel,GM702B_rel,"
                    "temperature,humidity,treatment,timestamp,time\n"
                )
            log.write_line(f"✅ Created: {os.path.abspath(READING_CSV_FILE)}")
        else:
            log.write_line(f"📄 Exists: {os.path.abspath(READING_CSV_FILE)}")

    def is_serial_port_available(self, port):
        try:
            with serial.Serial(port, BAUD_RATE, timeout=1) as test_ser:
                return True
        except serial.SerialException:
            return False
    
    def on_mount(self) -> None:
        # Connect to Log
        log = self.query_one("#reading_log", Log)

        # Check if the CSV file exists, create it if not
        log.write_line(str(os.getcwd()))
        self.init_csv_files()

        # Initialize panels
        self.query_one("#left_panel").border_title = "Control"
        self.query_one("#right_panel").border_title = "Log"
        self.query_one("#treatment_input").border_title = "Treatment Name"
        self.query_one("#run_input").border_title = "Number of Runs"
        self.query_one("#run_input", Input).value = "0"
        self.query_one("#plot_box").border_title = "Principal Component Analysis"
        self.query_one("#treatment_panel").border_title = "Treatments"
        self.update_plot()
        
        # Check ports
        if SERIAL_PORT:
            if self.is_serial_port_available(SERIAL_PORT):
                log.write_line(f"✅ Serial port available: {SERIAL_PORT}")
            else:
                log.write_line(f"❌ Serial port {SERIAL_PORT} is currently in use or unavailable.")
        else:
            log.write_line("❌ No serial port found. Please check your connection.")
    

    def send_command(self, ser, command):
        log = self.query_one("#reading_log", Log)
        """Sends a command to the serial device."""
        try:
            if ser and ser.is_open:
                ser.write((command + '\n').encode())  # Ensure newline for command
                log.write_line(f">>> Sent command: {command}")
            else:
                log.write_line("Serial port not open.")
        except Exception as e:
            log.write_line(f"Error sending command: {e}")


    def read_serial(self, treatment=None, record_control_air=False, timestamp=None):
        log = self.query_one("#reading_log", Log)
        log.write_line("Starting serial read...")
        global ser
        try:
            def open_serial():
                global ser
                if ser and ser.is_open:
                    ser.close()
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                time.sleep(2)
                log.write_line(f"Reconnected to Serial Port: {SERIAL_PORT}")

            open_serial()

            self.send_command(ser, "PUMP1 OFF")
            self.send_command(ser, "PUMP2 OFF")

            # === STEP 1: Control Air ===
            log.write_line("Step 1: Running control air (PUMP2 ON) and collecting 300 readings...")
            self.send_command(ser, "PUMP2 ON")
            control_readings = []
            control_count = 0
            last_valid_time = time.time()

            while control_count < 300 and not self.stop_flag.is_set():
                line = ser.readline().decode(errors='replace').strip()
                now = time.time()

                if line:
                    try:
                        values = list(map(float, line.split(',')))
                        if len(values) == 6:
                            control_readings.append(values)
                            control_count += 1
                            last_valid_time = now
                            log.write_line(f"Control {control_count}/300: {values}")
                    except ValueError:
                        continue

                if now - last_valid_time > 5:
                    log.write_line("⚠ No signal for 5 seconds. Reopening serial...")
                    open_serial()
                    self.send_command(ser, "PUMP2 ON")
                    last_valid_time = now

                time.sleep(1)

            self.send_command(ser, "PUMP2 OFF")

            if not control_readings:
                log.write_line("No control readings collected.")
                return

            # Process control data
            control_array = np.array(control_readings)
            control_mean = np.mean(control_array, axis=0)
            control_max = np.max(control_array, axis=0)
            control_min = np.min(control_array, axis=0)
            control_sd = np.std(control_array, axis=0, ddof=1)
            final_values = control_array[-1]
            correction_coeffs = self.get_env_correction_coefficients()

            if record_control_air:
                with open(CONTROL_CSV_FILE, 'a') as f:
                    row = list(control_mean) + list(control_max) + list(control_min) + list(control_sd) + list(final_values)
                    row = [f"{v:.4f}" for v in row]
                    row.append(treatment)
                    row.append(str(timestamp))
                    f.write(",".join(row) + "\n")

            # === STEP 2: Sample Air ===
            log.write_line("Step 2: Running sample air (PUMP1 ON) and collecting 100 readings...")
            self.send_command(ser, "PUMP1 ON")
            sample_readings = []
            sample_count = 0
            first_adjusted_gas = None
            last_valid_time = time.time()

            while sample_count < 100 and not self.stop_flag.is_set():
                line = ser.readline().decode(errors='replace').strip()
                now = time.time()

                if line:
                    values = line.split(',')
                    if len(values) == 6 and all(v.replace('.', '', 1).isdigit() for v in values):
                        values = list(map(float, values))
                        gas_values = values[:4]
                        temp_hum_values = values[4:]
                        temperature = temp_hum_values[0]
                        humidity = temp_hum_values[1]
                        sensor_labels = ['GM102B_mean', 'GM302B_mean', 'GM502B_mean', 'GM702B_mean']
                        adjusted_gas = []

                        for i, v in enumerate(gas_values):
                            f = final_values[i]
                            sensor = sensor_labels[i]
                            a, b, intercept = correction_coeffs.get(sensor, (0, 0, 0))
                            correction = a * temperature + b * humidity
                            baseline = f + correction
                            adjusted = v - baseline
                            adjusted_gas.append(adjusted)

                        if first_adjusted_gas is None:
                            first_adjusted_gas = adjusted_gas.copy()

                        normalized_gas = [val - base for val, base in zip(adjusted_gas, first_adjusted_gas)]
                        full_reading = normalized_gas + temp_hum_values
                        sample_readings.append(adjusted_gas)
                        sample_count += 1
                        last_valid_time = now

                        log.write_line(f"Sample {sample_count}/100: {full_reading}")
                        with open(READING_CSV_FILE, 'a') as f:
                            f.write(",".join(map(str, full_reading)) + f",{treatment},{timestamp},{sample_count}\n")
                        self.call_from_thread(self.update_plot)

                if now - last_valid_time > 5:
                    log.write_line("⚠ No signal for 5 seconds during sample. Reopening serial...")
                    open_serial()
                    self.send_command(ser, "PUMP1 ON")
                    last_valid_time = now

                time.sleep(1)

            self.send_command(ser, "PUMP1 OFF")
            self.send_command(ser, "PUMP2 OFF")
            run_input_widget = self.query_one("#run_input", Input)
            try:
                current_value = int(run_input_widget.value.strip())
                if current_value > 1:
                    run_input_widget.value = str(current_value - 1)
                else:
                    run_input_widget.value = "1"
            except ValueError:
                run_input_widget.value = "1"

        except Exception as e:
            log.write_line(f"Serial Read Error: {e}")
        finally:
            log.write_line("Cleaning up and closing serial port.")
            if ser and ser.is_open:
                ser.close()



    def get_env_correction_coefficients(self, control_csv=CONTROL_CSV_FILE):
        log = self.query_one("#reading_log", Log)
        try:
            df = pd.read_csv(control_csv)
            df = df.dropna()

            voc_sensors = ['GM102B_mean', 'GM302B_mean',
                        'GM502B_mean', 'GM702B_mean']
            temp_col = 'temperature_mean'
            hum_col = 'humidity_mean'

            coefficients = {}

            latest_treatment = df['treatment'].iloc[-1] if 'treatment' in df.columns else 'unknown'
            latest_timestamp = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else time.time()

            for sensor in voc_sensors:
                X = df[[temp_col, hum_col]]
                y = df[sensor]
                model = LinearRegression().fit(X, y)
                a = model.coef_[0]
                b = model.coef_[1]
                intercept = model.intercept_
                coefficients[sensor] = (a, b, intercept)

                log.write_line(f"{sensor} -> a: {a:.4f}, b: {b:.4f}, intercept: {intercept:.4f}")

                # Save to CSV
                with open(CORRECTION_CSV_FILE, 'a') as f:
                    f.write(
                        f"{sensor},{a:.6f},{b:.6f},{intercept:.6f},{latest_treatment},{latest_timestamp}\n")

            return coefficients

        except Exception as e:
            log.write_line(f"Error during regression analysis: {e}")
            return {}

    def run_enose_capture_template(self, treatment_input=None, record_control_air=False):
        self.stop_flag.clear()
        treatment = treatment_input
        timestamp = time.time()
        self.read_serial(treatment, record_control_air, timestamp)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        log = self.query_one("#reading_log", Log)
        run_button = self.query_one("#run_button", Button)

        if event.button.id == "run_button":
            if not self.reading_thread or not self.reading_thread.is_alive():
                # START: Begin new capture
                treatment_input = self.query_one("#treatment_input", Input).value.strip()
                run_input = self.query_one("#run_input", Input).value.strip()

                if treatment_input:
                    self.treatment_name = treatment_input
                try:
                    self.run_count = int(run_input)
                except ValueError:
                    self.run_count = 1

                log.write_line(f"▶ Starting e-nose with treatment: {self.treatment_name}, cycles: {self.run_count}")
                self.stop_flag.clear()
                self.pause_flag.clear()
                run_button.label = "Pause"

                self.reading_thread = threading.Thread(
                    target=self.run_enose_capture,
                    args=(self.treatment_name, False, self.run_count),
                    daemon=True
                )
                self.reading_thread.start()

            else:
                # TOGGLE: Pause or Resume
                if self.pause_flag.is_set():
                    self.pause_flag.clear()
                    run_button.label = "Pause"
                    log.write_line("▶ Resumed readings.")
                else:
                    self.pause_flag.set()
                    run_button.label = "Resume"  # ← previously "Start"
                    log.write_line("⏸ Paused readings.")

        elif event.button.id == "cancel_button":
            self.stop_flag.set()
            log.write_line("❌ Measurement schedule cancelled.")
            self.query_one("#run_button", Button).label = "Start"

        elif event.button.id == "exit_button":
            log.write_line("👋 Exiting program...")
            self.exit()
        
        elif event.button.id == "inc_button":
            run_input = self.query_one("#run_input", Input)
            try:
                val = int(run_input.value.strip() or "0")
                run_input.value = str(val + 1)
            except ValueError:
                run_input.value = "1"

        elif event.button.id == "dec_button":
            run_input = self.query_one("#run_input", Input)
            try:
                val = int(run_input.value.strip() or "0")
                run_input.value = str(max(1, val - 1))
            except ValueError:
                run_input.value = "1"

    def update_plot(self):
        shared_colors = [
            "red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"
        ]
        plot = self.query_one(PlotWidget)
        plot.clear()

        treatment_panel = self.query_one("#treatment_panel")
        treatment_panel.remove_children()  # Clear old labels

        try:
            df = pd.read_csv(READING_CSV_FILE)

            sensor_cols = ["GM102B_rel", "GM302B_rel", "GM502B_rel", "GM702B_rel"]
            if not all(col in df.columns for col in sensor_cols + ["treatment"]):
                self.query_one("#reading_log", Log).write_line("❌ Required columns missing in CSV.")
                return

            df = df.dropna(subset=sensor_cols + ["treatment"])

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df[sensor_cols])
            df["PC1"] = pca_result[:, 0]
            df["PC2"] = pca_result[:, 1]

            treatments = df["treatment"].unique()
            color_map = {treat: shared_colors[i % len(shared_colors)] for i, treat in enumerate(treatments)}

            for treat in treatments:
                group = df[df["treatment"] == treat]
                color = color_map[treat]
                plot.scatter(
                    group["PC1"].tolist(),
                    group["PC2"].tolist(),
                    marker="⦿",
                    marker_style=color,
                )
                label = Static(f"{treat}")
                label.styles.color = color
                label.styles.bold = True
                treatment_panel.mount(label)
            
            def get_symmetric_limits(values, margin_ratio=0.1):
                vmin = min(values)
                vmax = max(values)
                abs_max = max(abs(vmin), abs(vmax))
                limit = abs_max * (1 + margin_ratio)
                return -limit, limit

            xmin, xmax = get_symmetric_limits(df["PC1"])
            ymin, ymax = get_symmetric_limits(df["PC2"])

            plot.set_xlabel("PC1")
            plot.set_ylabel("PC2")
            plot.set_xlimits(xmin, xmax)
            plot.set_ylimits(ymin, ymax)
            plot.refresh()

        except Exception as e:
            self.query_one("#reading_log", Log).write_line(f"❌ Plot update error: {e}")
        
    def run_enose_capture(self, treatment_base_name="auto", record_control_air=False, num_cycles=1):
        log = self.query_one("#reading_log", Log)
        for cycle in range(num_cycles):
            # Generate unique treatment name using timestamp
            time_str = time.strftime("%Y%m%d_%H%M%S")
            treatment_name = f"{treatment_base_name}_{time_str}"
            log.write_line(f"\n>>> Starting capture {cycle + 1}/{num_cycles}: {treatment_name}")
            # Run one capture
            capture_thread = threading.Thread(
                target=self.run_enose_capture_template,
                args=(treatment_name, record_control_air),
            )
            capture_thread.start()
            capture_thread.join()  # Wait for the capture to complete

            # Countdown for next round (if not the last round)
            if cycle < num_cycles - 1:
                log.write_line("Waiting for next capture...")
                for i in range(30, 0, -1):
                    log.write_line(f"Next capture in {i} seconds")
                    time.sleep(1)

            if self.stop_flag.is_set():
                break
            log = self.query_one("#reading_log", Log)
            log.write_line(f"Cycle {cycle + 1}/{self.run_count}: collecting readings...")
            self.x_data = []
            self.y_data = []

            log.write_line(f"Cycle {cycle + 1} complete.\n")
            if cycle < self.run_count - 1 and not self.stop_flag.is_set():
                log.write_line("Waiting for next cycle...\n")
                for t in range(3):  # use short delay for demo
                    if self.stop_flag.is_set():
                        break
                    while self.pause_flag.is_set():
                        time.sleep(0.1)
                    time.sleep(1)
        if not self.stop_flag.is_set():
            self.query_one("#reading_log", Log).write("✅ All cycles complete.")
            self.query_one("#run_button", Button).label = "Start"

if __name__ == "__main__":
    ENoseApp().run()