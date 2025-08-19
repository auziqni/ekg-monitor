# filename: main.py
"""
Real-time EKG monitor for STM32 comma-separated + semicolon-terminated format.

- Serial format per sample:
  "t_ms_hex,CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8,CH9;"
  where:
    - t_ms_hex: 6-hex-digit milliseconds timestamp (wraps at ~0xFFFFFF)
    - CH1..CH9: 3-hex-digit values per channel (PB1, PB0, A7, A6, A5, A4, A3, A2, A1)
  Example:
    "012C3F,7FF,800,7F0,7E8,7E0,7D8,7D0,7C8,7C0;"

- Firmware debug line (logged as-is):
  "# debug_info -> sps: <sps>, target: <target>, efficiency: <eff>%"

This app adapts the timestamp monitor UI to:
  - Parse the new format using ';' as frame terminator
  - Show only one selected channel (1-9) in dual plots (raw / LMS filtered)
  - Keep LMS filter controls, logging, CSV recording, and performance labels
  - Compute local SPS while also logging firmware debug info
"""

import sys
import time
import threading
from collections import deque
import os
import csv

import numpy as np
import serial
import serial.tools.list_ports

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QComboBox,
    QPushButton,
    QLabel,
    QStatusBar,
    QGroupBox,
    QTextEdit,
    QSplitter,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont, QTextCursor

import pyqtgraph as pg


class LMSFilter:
    """Adaptive LMS Predictor Filter for EKG signal processing."""

    def __init__(self, filter_order: int = 20, step_size: float = 0.01) -> None:
        self.filter_order = filter_order
        self.step_size = step_size
        self.reset()

    def reset(self) -> None:
        """Reset filter coefficients and input buffer."""
        self.weights = np.zeros(self.filter_order)
        self.input_buffer = np.zeros(self.filter_order)
        self.sample_count = 0

    def update_parameters(self, filter_order: int, step_size: float) -> None:
        """Update filter parameters and reset if order changed."""
        if filter_order != self.filter_order:
            self.filter_order = filter_order
            self.reset()
        self.step_size = step_size

    def process_sample(self, input_sample: float) -> float:
        """Process single sample through LMS predictor filter."""
        self.sample_count += 1

        # Normalize input to prevent numerical instability (12-bit range)
        normalized_input = input_sample / 4095.0

        # Shift input buffer (FIFO)
        self.input_buffer[1:] = self.input_buffer[:-1]
        self.input_buffer[0] = normalized_input

        # Warm-up: return raw until we have enough history
        if self.sample_count < self.filter_order:
            return float(input_sample)

        # Predict current sample using previous samples
        predicted_normalized = float(np.dot(self.weights, self.input_buffer))

        # LMS update with input power normalization
        error = normalized_input - predicted_normalized
        input_power = float(np.dot(self.input_buffer, self.input_buffer)) + 1e-6
        normalized_step = self.step_size / input_power
        self.weights += normalized_step * error * self.input_buffer

        # Clip weights to prevent instability
        np.clip(self.weights, -10.0, 10.0, out=self.weights)

        # Convert back to original scale and clip output range
        predicted = predicted_normalized * 4095.0
        return float(np.clip(predicted, 0.0, 4095.0))


class SerialWorkerSTM(QObject):
    """Worker thread for STM comma-separated + semicolon-terminated stream."""

    data_received = pyqtSignal(list)  # [timestamp_seconds: float, value: int]
    info_received = pyqtSignal(dict)  # {'sps_fw': int, 'target': int, 'efficiency': int}
    status_changed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.serial_port: serial.Serial | None = None
        self.is_running: bool = False
        self.port_name: str = ""
        self.baudrate: int = 250000
        self._buffer = bytearray()

        # Local SPS computation
        self._local_sps_count = 0
        self._local_sps_last_ts = time.time()

    def connect_serial(self, port_name: str, baudrate: int) -> bool:
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()

            self.port_name = port_name
            self.baudrate = baudrate
            self.serial_port = serial.Serial(port_name, baudrate, timeout=0.1)
            self.status_changed.emit(f"Connected to {port_name}")
            return True
        except Exception as e:
            self.status_changed.emit(f"Connection failed: {str(e)}")
            return False

    def disconnect_serial(self) -> None:
        self.is_running = False
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
        finally:
            self.status_changed.emit("Disconnected")

    def start_reading(self) -> None:
        self.is_running = True
        self._buffer = bytearray()
        self._local_sps_count = 0
        self._local_sps_last_ts = time.time()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self) -> None:
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                # Read whatever is available; keep CPU usage reasonable
                waiting = self.serial_port.in_waiting
                chunk = self.serial_port.read(waiting or 1)
                if not chunk:
                    continue
                self._buffer.extend(chunk)

                # Process complete messages terminated by ';' (data) or '\n' (debug)
                while True:
                    semi_pos = self._buffer.find(b";")
                    nl_pos = self._buffer.find(b"\n")

                    # No terminator found
                    if semi_pos == -1 and nl_pos == -1:
                        break

                    # Choose earliest terminator
                    choose_nl = (nl_pos != -1) and (semi_pos == -1 or nl_pos < semi_pos)

                    if choose_nl:
                        frame = self._buffer[:nl_pos]
                        # Drop '\n' and optional preceding '\r'
                        self._buffer = self._buffer[nl_pos + 1 :]
                        if frame.endswith(b"\r"):
                            frame = frame[:-1]

                        if not frame:
                            continue

                        # Treat as text/debug line
                        text = frame.decode(errors="ignore").strip()
                        if not text:
                            continue
                        if text.startswith('#'):
                            info = self._parse_debug_line(text)
                            if info:
                                self.info_received.emit(info)
                            # Always log the line
                            self.status_changed.emit(text)
                        else:
                            # Non-data, non-debug: log raw
                            self.status_changed.emit(text)
                        continue

                    # Data frame terminated by ';'
                    frame = self._buffer[:semi_pos]
                    self._buffer = self._buffer[semi_pos + 1 :]
                    if not frame:
                        continue
                    self._parse_and_emit_data_frame(frame)

                # Emit local SPS roughly every second
                now = time.time()
                if now - self._local_sps_last_ts >= 1.0:
                    self.info_received.emit({"sps_local": self._local_sps_count})
                    self._local_sps_count = 0
                    self._local_sps_last_ts = now

            except Exception as e:
                if self.is_running:
                    self.status_changed.emit(f"Read error: {str(e)}")
                break

    def _parse_debug_line(self, line: str) -> dict:
        """Parse firmware debug line: '# debug_info -> sps: X, target: Y, efficiency: Z%'."""
        try:
            line = line.strip()
            if not line.startswith("#"):
                return {}
            # Very tolerant parsing
            # Extract integers in order: sps, target, efficiency
            import re

            sps_match = re.search(r"sps\s*:\s*(\d+)", line, re.IGNORECASE)
            tgt_match = re.search(r"target\s*:\s*(\d+)", line, re.IGNORECASE)
            eff_match = re.search(r"efficiency\s*:\s*(\d+)%", line, re.IGNORECASE)

            info = {}
            if sps_match:
                info["sps_fw"] = int(sps_match.group(1))
            if tgt_match:
                info["target"] = int(tgt_match.group(1))
            if eff_match:
                info["efficiency"] = int(eff_match.group(1))
            return info
        except Exception:
            return {}

    def _parse_and_emit_data_frame(self, frame_bytes: bytes) -> None:
        try:
            parts = frame_bytes.decode("ascii", errors="ignore").split(",")
            if len(parts) != 10:
                # Unexpected field count; log minimal info for diagnostics
                self.status_changed.emit(f"Malformed frame (fields={len(parts)}): {parts[:3]}...")
                return

            # Timestamp is hex milliseconds
            t_ms_hex = parts[0].strip()
            t_ms = int(t_ms_hex, 16)
            t_seconds = t_ms / 1000.0

            # 9 channels hex -> int
            ch_values = []
            for i in range(1, 10):
                ch_values.append(int(parts[i].strip(), 16))

            # Emit one data point per selected channel (selection handled in GUI)
            # For efficiency, we send the full vector; GUI picks the channel
            self.data_received.emit([t_seconds, ch_values])

            self._local_sps_count += 1
        except Exception as e:
            self.status_changed.emit(f"Frame parse error: {str(e)} | raw={frame_bytes[:32]!r}")


class EKGMonitorSTM(QMainWindow):
    """Main application window for STM stream with selectable channel and LMS filtering."""

    CHANNEL_LABELS = ["CH1: PB1", "CH2: PB0", "CH3: A7", "CH4: A6", "CH5: A5", "CH6: A4", "CH7: A3", "CH8: A2", "CH9: A1"]

    def __init__(self) -> None:
        super().__init__()
        self._setupLMSFilter()
        self._setupUI()
        self._setupSerial()
        self._setupData()
        self._setupTimer()
        self._setupRecording()

    def _setupLMSFilter(self) -> None:
        self.lms_filter = LMSFilter(filter_order=20, step_size=0.01)
        self.filter_enabled = True

    def _setupRecording(self) -> None:
        self.recording = False
        self.recording_duration = 60
        self.recording_remaining = 0
        self.csv_file = None
        self.csv_writer = None
        self.csv_buffer: list[list[str | int | float]] = []

        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self._update_recording_timer)

        if not os.path.exists("log"):
            os.makedirs("log")

    def _setupUI(self) -> None:
        self.setWindowTitle("EKG STM Monitor (CSV+; delim) with LMS Filter v1.0")
        self.setGeometry(100, 100, 1400, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control Panel
        control_group = QGroupBox("Control Panel")
        control_layout = QHBoxLayout(control_group)

        control_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(180)
        control_layout.addWidget(self.port_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_ports)
        control_layout.addWidget(self.refresh_btn)

        control_layout.addWidget(QLabel("Baudrate:"))
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "115200", "230400", "250000", "460800", "921600", "1000000"])
        self.baudrate_combo.setCurrentText("250000")
        control_layout.addWidget(self.baudrate_combo)

        control_layout.addWidget(QLabel("Channel (1-9):"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems([str(i) for i in range(1, 10)])
        self.channel_combo.setCurrentText("9")  # Default to CH9 (A1)
        control_layout.addWidget(self.channel_combo)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._toggle_connection)
        control_layout.addWidget(self.connect_btn)

        self.reset_btn = QPushButton("Reset Signal")
        self.reset_btn.clicked.connect(self._reset_signal)
        control_layout.addWidget(self.reset_btn)

        self.log_btn = QPushButton("Hide Log")
        self.log_btn.clicked.connect(self._toggle_log)
        control_layout.addWidget(self.log_btn)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self._toggle_recording)
        self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.record_btn.setEnabled(False)
        control_layout.addWidget(self.record_btn)

        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        control_layout.addWidget(self.status_label)

        control_layout.addStretch()
        main_layout.addWidget(control_group)

        # Filter Control Panel
        filter_group = QGroupBox("LMS Filter Control")
        filter_layout = QHBoxLayout(filter_group)

        self.filter_enable_cb = QCheckBox("Enable Filter")
        self.filter_enable_cb.setChecked(True)
        self.filter_enable_cb.stateChanged.connect(self._toggle_filter)
        filter_layout.addWidget(self.filter_enable_cb)

        filter_layout.addWidget(QLabel("Order:"))
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(5, 50)
        self.filter_order_spin.setValue(20)
        self.filter_order_spin.valueChanged.connect(self._update_filter_params)
        filter_layout.addWidget(self.filter_order_spin)
        order_info = QLabel("(5-50)")
        order_info.setStyleSheet("font-size: 9px; color: gray;")
        filter_layout.addWidget(order_info)

        filter_layout.addWidget(QLabel("Step Size (μ):"))
        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(0.001, 0.1)
        self.step_size_spin.setValue(0.01)
        self.step_size_spin.setDecimals(3)
        self.step_size_spin.setSingleStep(0.001)
        self.step_size_spin.valueChanged.connect(self._update_filter_params)
        filter_layout.addWidget(self.step_size_spin)
        step_info = QLabel("(0.001-0.1)")
        step_info.setStyleSheet("font-size: 9px; color: gray;")
        filter_layout.addWidget(step_info)

        self.reset_filter_btn = QPushButton("Reset Filter")
        self.reset_filter_btn.clicked.connect(self._reset_filter)
        filter_layout.addWidget(self.reset_filter_btn)

        filter_layout.addStretch()
        main_layout.addWidget(filter_group)

        # Performance metrics panel
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QHBoxLayout(metrics_group)

        self.sps_label = QLabel("SPS (local): --")
        self.sps_fw_label = QLabel("SPS (fw): --")
        self.efficiency_label = QLabel("Efficiency (fw): --%")
        self.target_label = QLabel("Target (fw): --")

        metrics_layout.addWidget(self.sps_label)
        metrics_layout.addWidget(self.sps_fw_label)
        metrics_layout.addWidget(self.efficiency_label)
        metrics_layout.addWidget(self.target_label)
        metrics_layout.addStretch()
        main_layout.addWidget(metrics_group)

        # Splitter: plots + log
        splitter = QSplitter(Qt.Vertical)
        self.splitter = splitter

        # Raw plot
        self.plot_widget_raw = pg.PlotWidget()
        self.plot_widget_raw.setLabel('left', 'Amplitude', units='')
        self.plot_widget_raw.setLabel('bottom', 'Time', units='s')
        self.plot_widget_raw.setTitle('EKG Signal - Raw (10 second window)')
        self.plot_widget_raw.setYRange(0, 4095)
        self.plot_widget_raw.showGrid(x=True, y=True)
        self.plot_widget_raw.setBackground('white')
        self.plot_line_raw = self.plot_widget_raw.plot(pen=pg.mkPen(color='red', width=2))
        splitter.addWidget(self.plot_widget_raw)

        # Filtered plot
        self.plot_widget_filtered = pg.PlotWidget()
        self.plot_widget_filtered.setLabel('left', 'Amplitude', units='')
        self.plot_widget_filtered.setLabel('bottom', 'Time', units='s')
        self.plot_widget_filtered.setTitle('EKG Signal - LMS Filtered (10 second window)')
        self.plot_widget_filtered.setYRange(0, 4095)
        self.plot_widget_filtered.showGrid(x=True, y=True)
        self.plot_widget_filtered.setBackground('white')
        self.plot_line_filtered = self.plot_widget_filtered.plot(pen=pg.mkPen(color='blue', width=2))
        splitter.addWidget(self.plot_widget_filtered)

        # Log area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(160)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background-color: #f0f0f0;")
        splitter.addWidget(self.log_text)
        self.log_visible = True

        splitter.setSizes([320, 320, 160])
        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - STM CSV mode")

        # Initial ports
        self._refresh_ports()

    def _setupSerial(self) -> None:
        self.serial_worker = SerialWorkerSTM()
        self.serial_worker.data_received.connect(self._process_data)
        self.serial_worker.info_received.connect(self._process_info)
        self.serial_worker.status_changed.connect(self._update_status)
        self.is_connected = False

    def _setupData(self) -> None:
        self.window_seconds = 10
        # Default buffer for 10s window; firmware example target ~400 fps
        self.max_samples = 4000
        self.time_data = deque(maxlen=self.max_samples)
        self.signal_data_raw = deque(maxlen=self.max_samples)
        self.signal_data_filtered = deque(maxlen=self.max_samples)
        self.current_performance = {"sps_local": 0}

        # Local SPS computation on GUI side too (redundant safety)
        self._gui_sps_count = 0
        self._gui_sps_last_ts = time.time()

    def _setupTimer(self) -> None:
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plot)

    # --- UI Actions ---
    def _refresh_ports(self) -> None:
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")

    def _toggle_connection(self) -> None:
        if not self.is_connected:
            if self.port_combo.currentText():
                port_name = self.port_combo.currentText().split(' - ')[0]
                baudrate = int(self.baudrate_combo.currentText())
                if self.serial_worker.connect_serial(port_name, baudrate):
                    self.serial_worker.start_reading()
                    self.is_connected = True
                    self.connect_btn.setText("Disconnect")
                    self.status_label.setText("Connected")
                    self.status_label.setStyleSheet("color: green; font-weight: bold;")
                    self.update_timer.start(50)
                    self.record_btn.setEnabled(True)
                    self.port_combo.setEnabled(False)
                    self.baudrate_combo.setEnabled(False)
                    self.refresh_btn.setEnabled(False)
                    self.channel_combo.setEnabled(False)
        else:
            self.serial_worker.disconnect_serial()
            self.is_connected = False
            self.connect_btn.setText("Connect")
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.update_timer.stop()
            if self.recording:
                self._stop_recording()
            self.record_btn.setEnabled(False)
            self.port_combo.setEnabled(True)
            self.baudrate_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self.channel_combo.setEnabled(True)

    def _toggle_log(self) -> None:
        if self.log_visible:
            self.log_text.hide()
            self.log_btn.setText("Show Log")
            self.log_visible = False
            self.splitter.setSizes([480, 480, 0])
        else:
            self.log_text.show()
            self.log_btn.setText("Hide Log")
            self.log_visible = True
            self.splitter.setSizes([320, 320, 160])

    def _toggle_filter(self, state: int) -> None:
        self.filter_enabled = state == Qt.Checked
        self._log_message(f"LMS Filter {'enabled' if self.filter_enabled else 'disabled'}")

    def _update_filter_params(self) -> None:
        order = self.filter_order_spin.value()
        step = self.step_size_spin.value()
        self.lms_filter.update_parameters(order, step)
        self._log_message(f"Filter params updated: Order={order}, μ={step}")

    def _reset_filter(self) -> None:
        self.lms_filter.reset()
        self._log_message("LMS Filter reset")

    def _reset_signal(self) -> None:
        self.time_data.clear()
        self.signal_data_raw.clear()
        self.signal_data_filtered.clear()
        self.plot_line_raw.setData([], [])
        self.plot_line_filtered.setData([], [])
        self.log_text.clear()
        self._gui_sps_count = 0
        self._gui_sps_last_ts = time.time()
        self.status_bar.showMessage("Signal reset - STM CSV mode")
        self._log_message("Signal data reset")
        self.lms_filter.reset()

    # --- Recording ---
    def _toggle_recording(self) -> None:
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        try:
            timestamp = time.strftime("%H-%M-%S")
            filename = f"log/stm_{timestamp}.csv"
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp_seconds', 'raw_value', 'filtered_value', 'channel'])
            self.recording = True
            self.recording_remaining = self.recording_duration
            self.record_btn.setText(f"{self.recording_remaining}s")
            self.record_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
            self.recording_timer.start(1000)
            self._log_message(f"Recording started: {filename}")
        except Exception as e:
            self._log_message(f"Recording start failed: {str(e)}")

    def _stop_recording(self) -> None:
        if self.recording:
            self.recording = False
            self.recording_timer.stop()
            # Flush any buffered rows and close file
            try:
                if self.csv_writer and self.csv_buffer:
                    self.csv_writer.writerows(self.csv_buffer)
                    self.csv_file.flush()
            except Exception:
                pass
            try:
                if self.csv_file:
                    self.csv_file.close()
            finally:
                self.csv_file = None
                self.csv_writer = None
                self.csv_buffer.clear()
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
            self._log_message("Recording stopped")

    def _update_recording_timer(self) -> None:
        if self.recording:
            self.recording_remaining -= 1
            if self.recording_remaining <= 0:
                self._stop_recording()
                self._log_message("Recording completed (60 seconds)")
            else:
                self.record_btn.setText(f"{self.recording_remaining}s")

    # --- Data processing ---
    def _process_data(self, payload: list) -> None:
        # payload: [timestamp_seconds: float, ch_values: list[int]]
        t_seconds, ch_values = payload

        # Select channel (1-9) -> index 0-8
        try:
            ch_index = int(self.channel_combo.currentText()) - 1
        except Exception:
            ch_index = 0
        ch_index = max(0, min(8, ch_index))

        if ch_index >= len(ch_values):
            return

        raw_value = int(ch_values[ch_index])

        # Store raw
        self.time_data.append(float(t_seconds))
        self.signal_data_raw.append(raw_value)

        # LMS filtered
        if self.filter_enabled:
            filtered_value = self.lms_filter.process_sample(raw_value)
        else:
            filtered_value = float(raw_value)

        self.signal_data_filtered.append(filtered_value)

        # Local GUI-side SPS (for additional confidence)
        self._gui_sps_count += 1
        now = time.time()
        if now - self._gui_sps_last_ts >= 1.0:
            self.sps_label.setText(f"SPS (local): {self._gui_sps_count}")
            self._gui_sps_count = 0
            self._gui_sps_last_ts = now

        # CSV buffering (batch writes for performance)
        if self.recording and self.csv_writer:
            try:
                self.csv_buffer.append([f"{t_seconds:.6f}", raw_value, f"{filtered_value:.3f}", ch_index + 1])
                if len(self.csv_buffer) >= 50:
                    self.csv_writer.writerows(self.csv_buffer)
                    self.csv_file.flush()
                    self.csv_buffer.clear()
            except Exception as e:
                self._log_message(f"CSV write error: {str(e)}")

        # Occasional debug log to avoid spam
        if len(self.time_data) % 500 == 0:
            label = self.CHANNEL_LABELS[ch_index]
            self._log_message(f"Data: {len(self.time_data)} samples | {label} | Last: {raw_value}")

    def _process_info(self, info: dict) -> None:
        # Update labels and log both firmware and local SPS if available
        if 'sps_local' in info:
            self.sps_label.setText(f"SPS (local): {info['sps_local']}")
        if 'sps_fw' in info:
            self.sps_fw_label.setText(f"SPS (fw): {info['sps_fw']}")
        if 'efficiency' in info:
            self.efficiency_label.setText(f"Efficiency (fw): {info['efficiency']}%")
        if 'target' in info:
            self.target_label.setText(f"Target (fw): {info['target']}")

        # Log concise summary
        summary = []
        for key in ("sps_local", "sps_fw", "target", "efficiency"):
            if key in info:
                summary.append(f"{key}={info[key]}")
        if summary:
            self._log_message("Info: " + ", ".join(summary))

    def _update_plot(self) -> None:
        if len(self.time_data) == 0:
            return

        time_array = np.array(self.time_data)
        raw_array = np.array(self.signal_data_raw)
        filtered_array = np.array(self.signal_data_filtered)

        self.plot_line_raw.setData(time_array, raw_array)
        self.plot_line_filtered.setData(time_array, filtered_array)

        current_time = float(time_array[-1])
        x_min = current_time - self.window_seconds
        x_max = current_time + 0.5
        self.plot_widget_raw.setXRange(x_min, x_max)
        self.plot_widget_filtered.setXRange(x_min, x_max)

        # Status bar concise summary
        ch_idx = int(self.channel_combo.currentText())
        self.status_bar.showMessage(
            f"Samples: {len(self.signal_data_raw)} | CH:{ch_idx} | Local SPS: {self._gui_sps_count} | Last t={current_time:.2f}s"
        )

    # --- Logging & status ---
    def _log_message(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def _update_status(self, message: str) -> None:
        self._log_message(f"Status: {message}")

    # --- Qt lifecycle ---
    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.recording:
            self._stop_recording()
        if self.is_connected:
            self.serial_worker.disconnect_serial()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = EKGMonitorSTM()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


