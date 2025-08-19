# filename: main_mock.py
"""
Mock playback EKG monitor for STM-format CSV files.

Reads CSV generated in STM format (per line):
  t_ms_hex,CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8,CH9;

Streams samples according to timestamps to simulate realtime (approx 400 SPS),
displays one selected channel (1-9) with raw and LMS-filtered plots, supports
logging and optional manual CSV recording (60s).
"""

import sys
import time
import threading
from collections import deque
import os
import csv

import numpy as np

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
    QFileDialog,
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont, QTextCursor

import pyqtgraph as pg


class LMSFilter:
    def __init__(self, filter_order: int = 20, step_size: float = 0.01) -> None:
        self.filter_order = filter_order
        self.step_size = step_size
        self.reset()

    def reset(self) -> None:
        self.weights = np.zeros(self.filter_order)
        self.input_buffer = np.zeros(self.filter_order)
        self.sample_count = 0

    def update_parameters(self, filter_order: int, step_size: float) -> None:
        if filter_order != self.filter_order:
            self.filter_order = filter_order
            self.reset()
        self.step_size = step_size

    def process_sample(self, input_sample: float) -> float:
        self.sample_count += 1

        normalized_input = input_sample / 4095.0
        self.input_buffer[1:] = self.input_buffer[:-1]
        self.input_buffer[0] = normalized_input

        if self.sample_count < self.filter_order:
            return float(input_sample)

        predicted_normalized = float(np.dot(self.weights, self.input_buffer))
        error = normalized_input - predicted_normalized
        input_power = float(np.dot(self.input_buffer, self.input_buffer)) + 1e-6
        normalized_step = self.step_size / input_power
        self.weights += normalized_step * error * self.input_buffer
        np.clip(self.weights, -10.0, 10.0, out=self.weights)
        predicted = predicted_normalized * 4095.0
        return float(np.clip(predicted, 0.0, 4095.0))


class FileMockWorker(QObject):
    data_received = pyqtSignal(list)  # [timestamp_seconds: float, ch_values: list[int]]
    info_received = pyqtSignal(dict)  # {'sps_local': int}
    status_changed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.is_running: bool = False
        self.file_path: str = ""
        self.playback_speed: float = 1.0

        self._local_sps_count = 0
        self._local_sps_last_ts = time.time()

    def start_playback(self, file_path: str, speed: float = 1.0) -> None:
        self.file_path = file_path
        self.playback_speed = max(0.1, float(speed))
        self.is_running = True
        self._local_sps_count = 0
        self._local_sps_last_ts = time.time()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def stop(self) -> None:
        self.is_running = False

    def _play_loop(self) -> None:
        try:
            with open(self.file_path, 'r', encoding='ascii', newline='') as f:
                prev_t: float | None = None
                start_wall: float | None = None
                first_t: float | None = None

                for raw_line in f:
                    if not self.is_running:
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        # Log debug lines if any
                        self.status_changed.emit(line)
                        continue
                    if not line.endswith(';'):
                        continue

                    try:
                        payload = line[:-1]  # remove ';'
                        parts = payload.split(',')
                        if len(parts) != 10:
                            continue
                        t_ms = int(parts[0].strip(), 16)
                        t_seconds = t_ms / 1000.0
                        ch_values = [int(p.strip(), 16) for p in parts[1:]]
                    except Exception:
                        continue

                    # Playback pacing based on timestamps (approx realtime)
                    if first_t is None:
                        first_t = t_seconds
                        start_wall = time.perf_counter()
                    else:
                        target_elapsed = (t_seconds - first_t) / self.playback_speed
                        now = time.perf_counter()
                        if start_wall is not None:
                            to_sleep = target_elapsed - (now - start_wall)
                            if to_sleep > 0:
                                time.sleep(min(to_sleep, 0.05))

                    self.data_received.emit([t_seconds, ch_values])
                    self._local_sps_count += 1

                    # Periodic SPS
                    now_wall = time.time()
                    if now_wall - self._local_sps_last_ts >= 1.0:
                        self.info_received.emit({"sps_local": self._local_sps_count})
                        self._local_sps_count = 0
                        self._local_sps_last_ts = now_wall
        except Exception as e:
            if self.is_running:
                self.status_changed.emit(f"Playback error: {str(e)}")


class EKGMonitorMock(QMainWindow):
    CHANNEL_LABELS = [
        "CH1: PB1",
        "CH2: PB0",
        "CH3: A7",
        "CH4: A6",
        "CH5: A5",
        "CH6: A4",
        "CH7: A3",
        "CH8: A2",
        "CH9: A1",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._setupLMSFilter()
        self._setupUI()
        self._setupWorker()
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
        if not os.path.exists('log'):
            os.makedirs('log')

    def _setupUI(self) -> None:
        self.setWindowTitle("EKG Mock Playback Monitor with LMS Filter v1.0")
        self.setGeometry(100, 100, 1400, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control Panel
        control_group = QGroupBox("Control Panel")
        control_layout = QHBoxLayout(control_group)

        self.open_btn = QPushButton("Open CSV")
        self.open_btn.clicked.connect(self._open_csv)
        control_layout.addWidget(self.open_btn)

        self.file_label = QLabel("No file selected")
        control_layout.addWidget(self.file_label)

        control_layout.addWidget(QLabel("Channel (1-9):"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems([str(i) for i in range(1, 10)])
        self.channel_combo.setCurrentText("9")
        control_layout.addWidget(self.channel_combo)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start_playback)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_playback)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

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

        self.status_label = QLabel("Stopped")
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
        metrics_layout.addWidget(self.sps_label)
        metrics_layout.addStretch()
        main_layout.addWidget(metrics_group)

        # Splitter: plots + log
        splitter = QSplitter(Qt.Vertical)
        self.splitter = splitter

        self.plot_widget_raw = pg.PlotWidget()
        self.plot_widget_raw.setLabel('left', 'Amplitude', units='')
        self.plot_widget_raw.setLabel('bottom', 'Time', units='s')
        self.plot_widget_raw.setTitle('EKG Signal - Raw (10 second window)')
        self.plot_widget_raw.setYRange(0, 4095)
        self.plot_widget_raw.showGrid(x=True, y=True)
        self.plot_widget_raw.setBackground('white')
        self.plot_line_raw = self.plot_widget_raw.plot(pen=pg.mkPen(color='red', width=2))
        splitter.addWidget(self.plot_widget_raw)

        self.plot_widget_filtered = pg.PlotWidget()
        self.plot_widget_filtered.setLabel('left', 'Amplitude', units='')
        self.plot_widget_filtered.setLabel('bottom', 'Time', units='s')
        self.plot_widget_filtered.setTitle('EKG Signal - LMS Filtered (10 second window)')
        self.plot_widget_filtered.setYRange(0, 4095)
        self.plot_widget_filtered.showGrid(x=True, y=True)
        self.plot_widget_filtered.setBackground('white')
        self.plot_line_filtered = self.plot_widget_filtered.plot(pen=pg.mkPen(color='blue', width=2))
        splitter.addWidget(self.plot_widget_filtered)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(160)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background-color: #f0f0f0;")
        splitter.addWidget(self.log_text)
        self.log_visible = True

        splitter.setSizes([320, 320, 160])
        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Mock playback mode")

    def _setupWorker(self) -> None:
        self.worker = FileMockWorker()
        self.worker.data_received.connect(self._process_data)
        self.worker.info_received.connect(self._process_info)
        self.worker.status_changed.connect(self._update_status)
        self.playing = False
        self.opened_file: str | None = None

    def _setupData(self) -> None:
        self.window_seconds = 10
        self.max_samples = 4000
        self.time_data = deque(maxlen=self.max_samples)
        self.signal_data_raw = deque(maxlen=self.max_samples)
        self.signal_data_filtered = deque(maxlen=self.max_samples)
        self._gui_sps_count = 0
        self._gui_sps_last_ts = time.time()

    def _setupTimer(self) -> None:
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plot)

    # --- UI Actions ---
    def _open_csv(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Mock CSV", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.opened_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.start_btn.setEnabled(True)
            self._log_message(f"Selected file: {file_path}")

    def _start_playback(self) -> None:
        if not self.opened_file or self.playing:
            return
        self.worker.start_playback(self.opened_file, speed=1.0)
        self.playing = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.record_btn.setEnabled(True)
        self.status_label.setText("Playing")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.update_timer.start(50)

    def _stop_playback(self) -> None:
        if not self.playing:
            return
        self.worker.stop()
        self.playing = False
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True if self.opened_file else False)
        self.record_btn.setEnabled(False)
        if self.recording:
            self._stop_recording()
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.update_timer.stop()

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
        self.status_bar.showMessage("Signal reset - Mock mode")
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
            filename = f"log/mock_{timestamp}.csv"
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
        t_seconds, ch_values = payload
        try:
            ch_index = int(self.channel_combo.currentText()) - 1
        except Exception:
            ch_index = 0
        ch_index = max(0, min(8, ch_index))
        if ch_index >= len(ch_values):
            return

        raw_value = int(ch_values[ch_index])
        self.time_data.append(float(t_seconds))
        self.signal_data_raw.append(raw_value)
        if self.filter_enabled:
            filtered_value = self.lms_filter.process_sample(raw_value)
        else:
            filtered_value = float(raw_value)
        self.signal_data_filtered.append(filtered_value)

        # GUI-side SPS (soft)
        self._gui_sps_count += 1
        now = time.time()
        if now - self._gui_sps_last_ts >= 1.0:
            self.sps_label.setText(f"SPS (local): {self._gui_sps_count}")
            self._gui_sps_count = 0
            self._gui_sps_last_ts = now

        # CSV buffering
        if self.recording and self.csv_writer:
            try:
                self.csv_buffer.append([f"{t_seconds:.6f}", raw_value, f"{filtered_value:.3f}", ch_index + 1])
                if len(self.csv_buffer) >= 50:
                    self.csv_writer.writerows(self.csv_buffer)
                    self.csv_file.flush()
                    self.csv_buffer.clear()
            except Exception as e:
                self._log_message(f"CSV write error: {str(e)}")

        # Occasional log
        if len(self.time_data) % 500 == 0:
            label = self.CHANNEL_LABELS[ch_index]
            self._log_message(f"Data: {len(self.time_data)} samples | {label} | Last: {raw_value}")

    def _process_info(self, info: dict) -> None:
        if 'sps_local' in info:
            self.sps_label.setText(f"SPS (local): {info['sps_local']}")

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
        ch_idx = int(self.channel_combo.currentText())
        self.status_bar.showMessage(
            f"Samples: {len(self.signal_data_raw)} | CH:{ch_idx} | Local SPS: {self._gui_sps_count} | Last t={current_time:.2f}s"
        )

    def _log_message(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def _update_status(self, message: str) -> None:
        self._log_message(f"Status: {message}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.recording:
            self._stop_recording()
        if self.playing:
            self.worker.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = EKGMonitorMock()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


