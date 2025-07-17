# filename: dualSignalMonitor.py
# revisi: v1.0  
# short description: Real-time dual channel EKG signal monitoring with hex CSV format,
#                   auto-save CSV recording (60s max), manual start/stop controls,
#                   10-second sliding window display, and adaptive LMS filtering
#                   for ESP32 + ADS1115. Features dual plot display (raw/filtered)
#                   with separate LMS predictor filters for CH0 and CH1.
# this file is for dualSignal-continuous.cpp

import sys
import time
import threading
import numpy as np
import serial
import serial.tools.list_ports
import os
import csv
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                           QHBoxLayout, QWidget, QComboBox, QPushButton, 
                           QLabel, QStatusBar, QGroupBox, QTextEdit, QSplitter,
                           QSpinBox, QDoubleSpinBox, QCheckBox, QGridLayout)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont, QTextCursor
import pyqtgraph as pg

class LMSFilter:
    """Adaptive LMS Predictor Filter for EKG signal processing"""
    
    def __init__(self, filter_order=20, step_size=0.0001):
        self.filter_order = filter_order
        self.step_size = step_size
        self.reset()
        
    def reset(self):
        """Reset filter coefficients and input buffer"""
        self.weights = np.zeros(self.filter_order)
        self.input_buffer = np.zeros(self.filter_order)
        self.initialized = False
        self.sample_count = 0
        
    def update_parameters(self, filter_order, step_size):
        """Update filter parameters and reset if order changed"""
        if filter_order != self.filter_order:
            self.filter_order = filter_order
            self.reset()
        self.step_size = step_size
        
    def process_sample(self, input_sample):
        """Process single sample through LMS predictor filter"""
        self.sample_count += 1
        
        # Normalize input to prevent numerical instability
        normalized_input = input_sample / 4095.0
        
        # Shift input buffer (FIFO)
        self.input_buffer[1:] = self.input_buffer[:-1]
        self.input_buffer[0] = normalized_input
        
        # If not enough samples yet, return original
        if self.sample_count < self.filter_order:
            return input_sample
            
        # Predict current sample using previous samples
        predicted_normalized = np.dot(self.weights, self.input_buffer)
        
        # Calculate prediction error
        error = normalized_input - predicted_normalized
        
        # Normalized LMS update with input power normalization
        input_power = np.dot(self.input_buffer, self.input_buffer) + 1e-6
        normalized_step = self.step_size / input_power
        
        # Update weights using normalized LMS algorithm
        self.weights += normalized_step * error * self.input_buffer
        
        # Clip weights to prevent instability
        self.weights = np.clip(self.weights, -10.0, 10.0)
        
        # Convert back to original scale
        predicted = predicted_normalized * 4095.0
        
        # Ensure output is within valid range
        predicted = np.clip(predicted, 0, 4095)
        
        return predicted

class DualSerialWorker(QObject):
    """Worker thread for dual channel hex CSV serial communication"""
    data_received = pyqtSignal(list)
    info_received = pyqtSignal(dict)
    warning_received = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_running = False
        self.port_name = ""
        self.baudrate = 250000
        self.last_timestamp = None
        self.sample_interval_us = 1163  # 1000000/860 for interpolation
        self.start_time = None
        self.current_sample_count = 0
        
    def connect_serial(self, port_name, baudrate):
        """Connect to serial port"""
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                
            self.port_name = port_name
            self.baudrate = baudrate
            self.serial_port = serial.Serial(port_name, baudrate, timeout=1)
            self.status_changed.emit(f"Connected to {port_name}")
            return True
        except Exception as e:
            self.status_changed.emit(f"Connection failed: {str(e)}")
            return False
    
    def disconnect_serial(self):
        """Disconnect from serial port"""
        self.is_running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.status_changed.emit("Disconnected")
    
    def start_reading(self):
        """Start reading data from serial port"""
        self.is_running = True
        self.last_timestamp = None
        self.current_sample_count = 0
        self.start_time = None
        threading.Thread(target=self._read_loop, daemon=True).start()
    
    def parse_info_line(self, line):
        """Parse info line: # TS:1000000 SPS:860 CH0:430 CH1:430 EFF:98% SKIP:0 TOT:5487"""
        info = {}
        try:
            parts = line[1:].strip().split()
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    if key == 'TS':
                        info['timestamp'] = int(value)
                        self.last_timestamp = info['timestamp']
                    elif key == 'SPS':
                        info['sps'] = int(value)
                    elif key == 'CH0':
                        info['ch0_sps'] = int(value)
                    elif key == 'CH1':
                        info['ch1_sps'] = int(value)
                    elif key == 'EFF':
                        info['efficiency'] = int(value.replace('%', ''))
                    elif key == 'SKIP':
                        info['skipped'] = int(value)
                    elif key == 'TOT':
                        info['total'] = int(value)
        except Exception as e:
            self.status_changed.emit(f"Info parse error: {str(e)}")
        return info
    
    def parse_dual_hex_csv(self, line):
        """Parse dual hex CSV line: [7FF,800],[801,7FE],[ABC,DEF],[123,456]"""
        try:
            # Find all [xxx,yyy] patterns
            import re
            pattern = r'\[([0-9A-Fa-f]+),([0-9A-Fa-f]+)\]'
            matches = re.findall(pattern, line)
            
            ch0_values = []
            ch1_values = []
            
            for match in matches:
                ch0_hex, ch1_hex = match
                ch0_values.append(int(ch0_hex, 16))
                ch1_values.append(int(ch1_hex, 16))
            
            return ch0_values, ch1_values
        except Exception as e:
            self.status_changed.emit(f"Dual CSV parse error: {str(e)}")
            return [], []
    
    def interpolate_timestamps(self, data_count):
        """Create interpolated timestamps for data points starting from 0"""
        timestamps = []
        for i in range(data_count):
            time_seconds = (self.current_sample_count + i) * self.sample_interval_us / 1000000.0
            timestamps.append(time_seconds)
            
        self.current_sample_count += data_count
        return timestamps
    
    def _read_loop(self):
        """Main reading loop for dual channel format"""
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                if not line:
                    continue
                    
                if line.startswith('#'):
                    # Info message
                    info = self.parse_info_line(line)
                    if info:
                        self.info_received.emit(info)
                        
                elif line.startswith('*'):
                    # Warning message
                    warning_msg = line[1:].strip()
                    self.warning_received.emit(warning_msg)
                    
                else:
                    # Data line (dual hex CSV)
                    ch0_values, ch1_values = self.parse_dual_hex_csv(line)
                    if ch0_values and ch1_values:
                        # Create timestamps for each value pair
                        timestamps = self.interpolate_timestamps(len(ch0_values))
                        # Emit data with timestamps
                        self.data_received.emit([timestamps, ch0_values, ch1_values])
                        
            except Exception as e:
                if self.is_running:
                    self.status_changed.emit(f"Read error: {str(e)}")
                break

class DualEKGMonitor(QMainWindow):
    """Main application window for dual channel monitoring with LMS filtering"""
    
    def __init__(self):
        super().__init__()
        self.setupLMSFilters()
        self.setupUI()
        self.setupSerial()
        self.setupData()
        self.setupTimer()
        self.setupRecording()
        
    def setupLMSFilters(self):
        """Setup dual LMS filters"""
        self.lms_filter_ch0 = LMSFilter(filter_order=20, step_size=0.01)
        self.lms_filter_ch1 = LMSFilter(filter_order=20, step_size=0.01)
        self.filter_enabled_ch0 = True
        self.filter_enabled_ch1 = True
        
    def setupRecording(self):
        """Setup CSV recording functionality"""
        self.recording = False
        self.recording_start_time = None
        self.recording_duration = 60
        self.recording_remaining = 0
        self.csv_file = None
        self.csv_writer = None
        
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_timer)
        
        if not os.path.exists('log'):
            os.makedirs('log')
        
    def setupUI(self):
        """Setup user interface"""
        self.setWindowTitle("Dual Channel EKG Monitor with LMS Filter v1.0")
        self.setGeometry(100, 100, 1600, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header control panel
        control_group = QGroupBox("Control Panel")
        control_layout = QHBoxLayout(control_group)
        
        # Port selection
        control_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(150)
        control_layout.addWidget(self.port_combo)
        
        # Refresh ports button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        control_layout.addWidget(self.refresh_btn)
        
        # Baudrate selection
        control_layout.addWidget(QLabel("Baudrate:"))
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "115200", "250000", "500000", "1000000"])
        self.baudrate_combo.setCurrentText("250000")
        control_layout.addWidget(self.baudrate_combo)
        
        # Connect button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        control_layout.addWidget(self.connect_btn)
        
        # Reset button
        self.reset_btn = QPushButton("Reset Signal")
        self.reset_btn.clicked.connect(self.reset_signal)
        control_layout.addWidget(self.reset_btn)
        
        # Show/Hide log button
        self.log_btn = QPushButton("Hide Log")
        self.log_btn.clicked.connect(self.toggle_log)
        control_layout.addWidget(self.log_btn)
        
        # Recording button
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.record_btn.setEnabled(False)
        control_layout.addWidget(self.record_btn)
        
        # Connection status
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_group)
        
        # Signal visibility control panel
        visibility_group = QGroupBox("Signal Visibility")
        visibility_layout = QHBoxLayout(visibility_group)
        
        # CH0 checkboxes
        self.ch0_raw_cb = QCheckBox("CH0 Raw")
        self.ch0_raw_cb.setChecked(True)
        self.ch0_raw_cb.stateChanged.connect(self.update_signal_visibility)
        visibility_layout.addWidget(self.ch0_raw_cb)
        
        self.ch0_filtered_cb = QCheckBox("CH0 Filtered")
        self.ch0_filtered_cb.setChecked(True)
        self.ch0_filtered_cb.stateChanged.connect(self.update_signal_visibility)
        visibility_layout.addWidget(self.ch0_filtered_cb)
        
        # CH1 checkboxes
        self.ch1_raw_cb = QCheckBox("CH1 Raw")
        self.ch1_raw_cb.setChecked(True)
        self.ch1_raw_cb.stateChanged.connect(self.update_signal_visibility)
        visibility_layout.addWidget(self.ch1_raw_cb)
        
        self.ch1_filtered_cb = QCheckBox("CH1 Filtered")
        self.ch1_filtered_cb.setChecked(True)
        self.ch1_filtered_cb.stateChanged.connect(self.update_signal_visibility)
        visibility_layout.addWidget(self.ch1_filtered_cb)
        
        visibility_layout.addStretch()
        main_layout.addWidget(visibility_group)
        
        # Dual Filter Control Panel
        filter_group = QGroupBox("Dual LMS Filter Control")
        filter_layout = QGridLayout(filter_group)
        
        # CH0 Filter controls
        filter_layout.addWidget(QLabel("CH0 Filter:"), 0, 0)
        self.filter_enable_ch0_cb = QCheckBox("Enable")
        self.filter_enable_ch0_cb.setChecked(True)
        self.filter_enable_ch0_cb.stateChanged.connect(self.toggle_filter_ch0)
        filter_layout.addWidget(self.filter_enable_ch0_cb, 0, 1)
        
        filter_layout.addWidget(QLabel("Order:"), 0, 2)
        self.filter_order_ch0_spin = QSpinBox()
        self.filter_order_ch0_spin.setRange(5, 50)
        self.filter_order_ch0_spin.setValue(20)
        self.filter_order_ch0_spin.valueChanged.connect(self.update_filter_params_ch0)
        filter_layout.addWidget(self.filter_order_ch0_spin, 0, 3)
        
        filter_layout.addWidget(QLabel("Step Size:"), 0, 4)
        self.step_size_ch0_spin = QDoubleSpinBox()
        self.step_size_ch0_spin.setRange(0.001, 0.1)
        self.step_size_ch0_spin.setValue(0.01)
        self.step_size_ch0_spin.setDecimals(3)
        self.step_size_ch0_spin.setSingleStep(0.001)
        self.step_size_ch0_spin.valueChanged.connect(self.update_filter_params_ch0)
        filter_layout.addWidget(self.step_size_ch0_spin, 0, 5)
        
        self.reset_filter_ch0_btn = QPushButton("Reset CH0")
        self.reset_filter_ch0_btn.clicked.connect(self.reset_filter_ch0)
        filter_layout.addWidget(self.reset_filter_ch0_btn, 0, 6)
        
        # CH1 Filter controls
        filter_layout.addWidget(QLabel("CH1 Filter:"), 1, 0)
        self.filter_enable_ch1_cb = QCheckBox("Enable")
        self.filter_enable_ch1_cb.setChecked(True)
        self.filter_enable_ch1_cb.stateChanged.connect(self.toggle_filter_ch1)
        filter_layout.addWidget(self.filter_enable_ch1_cb, 1, 1)
        
        filter_layout.addWidget(QLabel("Order:"), 1, 2)
        self.filter_order_ch1_spin = QSpinBox()
        self.filter_order_ch1_spin.setRange(5, 50)
        self.filter_order_ch1_spin.setValue(20)
        self.filter_order_ch1_spin.valueChanged.connect(self.update_filter_params_ch1)
        filter_layout.addWidget(self.filter_order_ch1_spin, 1, 3)
        
        filter_layout.addWidget(QLabel("Step Size:"), 1, 4)
        self.step_size_ch1_spin = QDoubleSpinBox()
        self.step_size_ch1_spin.setRange(0.001, 0.1)
        self.step_size_ch1_spin.setValue(0.01)
        self.step_size_ch1_spin.setDecimals(3)
        self.step_size_ch1_spin.setSingleStep(0.001)
        self.step_size_ch1_spin.valueChanged.connect(self.update_filter_params_ch1)
        filter_layout.addWidget(self.step_size_ch1_spin, 1, 5)
        
        self.reset_filter_ch1_btn = QPushButton("Reset CH1")
        self.reset_filter_ch1_btn.clicked.connect(self.reset_filter_ch1)
        filter_layout.addWidget(self.reset_filter_ch1_btn, 1, 6)
        
        main_layout.addWidget(filter_group)
        
        # Performance metrics panel
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QHBoxLayout(metrics_group)
        
        self.sps_label = QLabel("SPS: --")
        self.ch0_sps_label = QLabel("CH0: --")
        self.ch1_sps_label = QLabel("CH1: --")
        self.efficiency_label = QLabel("Efficiency: --%")
        self.skipped_label = QLabel("Skipped: --")
        self.total_label = QLabel("Total: --")
        
        metrics_layout.addWidget(self.sps_label)
        metrics_layout.addWidget(self.ch0_sps_label)
        metrics_layout.addWidget(self.ch1_sps_label)
        metrics_layout.addWidget(self.efficiency_label)
        metrics_layout.addWidget(self.skipped_label)
        metrics_layout.addWidget(self.total_label)
        metrics_layout.addStretch()
        
        main_layout.addWidget(metrics_group)
        
        # Splitter for plots and log
        splitter = QSplitter(Qt.Vertical)
        self.splitter = splitter
        
        # Raw signal plot widget
        self.plot_widget_raw = pg.PlotWidget()
        self.plot_widget_raw.setLabel('left', 'Amplitude', units='')
        self.plot_widget_raw.setLabel('bottom', 'Time', units='s')
        self.plot_widget_raw.setTitle('EKG Signal - Raw (10 second window)')
        self.plot_widget_raw.setYRange(0, 4095)
        self.plot_widget_raw.showGrid(x=True, y=True)
        self.plot_widget_raw.setBackground('white')
        
        # Create plot lines with legend
        self.plot_line_ch0_raw = self.plot_widget_raw.plot(pen=pg.mkPen(color='blue', width=2), name='CH0 Raw')
        self.plot_line_ch1_raw = self.plot_widget_raw.plot(pen=pg.mkPen(color='gold', width=2), name='CH1 Raw')
        self.plot_widget_raw.addLegend()
        
        splitter.addWidget(self.plot_widget_raw)
        
        # Filtered signal plot widget
        self.plot_widget_filtered = pg.PlotWidget()
        self.plot_widget_filtered.setLabel('left', 'Amplitude', units='')
        self.plot_widget_filtered.setLabel('bottom', 'Time', units='s')
        self.plot_widget_filtered.setTitle('EKG Signal - LMS Filtered (10 second window)')
        self.plot_widget_filtered.setYRange(0, 4095)
        self.plot_widget_filtered.showGrid(x=True, y=True)
        self.plot_widget_filtered.setBackground('white')
        
        # Create filtered plot lines with legend
        self.plot_line_ch0_filtered = self.plot_widget_filtered.plot(pen=pg.mkPen(color='blue', width=2), name='CH0 Filtered')
        self.plot_line_ch1_filtered = self.plot_widget_filtered.plot(pen=pg.mkPen(color='gold', width=2), name='CH1 Filtered')
        self.plot_widget_filtered.addLegend()
        
        splitter.addWidget(self.plot_widget_filtered)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background-color: #f0f0f0;")
        splitter.addWidget(self.log_text)
        self.log_visible = True
        
        # Set splitter proportions
        splitter.setSizes([300, 300, 150])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initial port refresh
        self.refresh_ports()
        
    def update_signal_visibility(self):
        """Update signal visibility based on checkboxes"""
        self.plot_line_ch0_raw.setVisible(self.ch0_raw_cb.isChecked())
        self.plot_line_ch1_raw.setVisible(self.ch1_raw_cb.isChecked())
        self.plot_line_ch0_filtered.setVisible(self.ch0_filtered_cb.isChecked())
        self.plot_line_ch1_filtered.setVisible(self.ch1_filtered_cb.isChecked())
        
    def toggle_filter_ch0(self, state):
        """Toggle CH0 filter enable/disable"""
        self.filter_enabled_ch0 = state == Qt.Checked
        status = "enabled" if self.filter_enabled_ch0 else "disabled"
        self.log_message(f"CH0 LMS Filter {status}")
        
    def toggle_filter_ch1(self, state):
        """Toggle CH1 filter enable/disable"""
        self.filter_enabled_ch1 = state == Qt.Checked
        status = "enabled" if self.filter_enabled_ch1 else "disabled"
        self.log_message(f"CH1 LMS Filter {status}")
        
    def update_filter_params_ch0(self):
        """Update CH0 filter parameters"""
        filter_order = self.filter_order_ch0_spin.value()
        step_size = self.step_size_ch0_spin.value()
        self.lms_filter_ch0.update_parameters(filter_order, step_size)
        self.log_message(f"CH0 Filter params updated: Order={filter_order}, μ={step_size}")
        
    def update_filter_params_ch1(self):
        """Update CH1 filter parameters"""
        filter_order = self.filter_order_ch1_spin.value()
        step_size = self.step_size_ch1_spin.value()
        self.lms_filter_ch1.update_parameters(filter_order, step_size)
        self.log_message(f"CH1 Filter params updated: Order={filter_order}, μ={step_size}")
        
    def reset_filter_ch0(self):
        """Reset CH0 filter coefficients"""
        self.lms_filter_ch0.reset()
        self.log_message("CH0 LMS Filter reset")
        
    def reset_filter_ch1(self):
        """Reset CH1 filter coefficients"""
        self.lms_filter_ch1.reset()
        self.log_message("CH1 LMS Filter reset")
        
    def reset_signal(self):
        """Reset signal data and plots"""
        self.time_data.clear()
        self.ch0_raw_data.clear()
        self.ch0_filtered_data.clear()
        self.ch1_raw_data.clear()
        self.ch1_filtered_data.clear()
        self.start_time = None
        
        self.plot_line_ch0_raw.setData([], [])
        self.plot_line_ch1_raw.setData([], [])
        self.plot_line_ch0_filtered.setData([], [])
        self.plot_line_ch1_filtered.setData([], [])
        
        self.log_text.clear()
        self.status_bar.showMessage("Signal reset")
        self.log_message("Signal data reset")
        
        # Reset both filters
        self.lms_filter_ch0.reset()
        self.lms_filter_ch1.reset()
        
    def setupSerial(self):
        """Setup serial worker"""
        self.serial_worker = DualSerialWorker()
        self.serial_worker.data_received.connect(self.process_data)
        self.serial_worker.info_received.connect(self.process_info)
        self.serial_worker.warning_received.connect(self.process_warning)
        self.serial_worker.status_changed.connect(self.update_status)
        self.is_connected = False
        
    def setupData(self):
        """Setup data storage"""
        self.window_seconds = 10
        self.max_samples = 8600  # 860 SPS * 10 seconds
        self.time_data = deque(maxlen=self.max_samples)
        self.ch0_raw_data = deque(maxlen=self.max_samples)
        self.ch0_filtered_data = deque(maxlen=self.max_samples)
        self.ch1_raw_data = deque(maxlen=self.max_samples)
        self.ch1_filtered_data = deque(maxlen=self.max_samples)
        self.start_time = None
        self.current_performance = {}
        
    def setupTimer(self):
        """Setup update timer"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        
    def toggle_recording(self):
        """Toggle CSV recording"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start CSV recording"""
        try:
            timestamp = time.strftime("%H-%M-%S")
            filename = f"log/dual_{timestamp}.csv"
            
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header with dual format
            self.csv_writer.writerow(['timestamp', 'ch0_raw', 'ch0_filtered', 'ch1_raw', 'ch1_filtered'])
            
            self.recording = True
            self.recording_start_time = time.time()
            self.recording_remaining = self.recording_duration
            
            self.record_btn.setText(f"{self.recording_remaining}s")
            self.record_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
            
            self.recording_timer.start(1000)
            
            self.log_message(f"Dual recording started: {filename}")
            
        except Exception as e:
            self.log_message(f"Recording start failed: {str(e)}")
            
    def stop_recording(self):
        """Stop CSV recording"""
        if self.recording:
            self.recording = False
            self.recording_timer.stop()
            
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
            
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
            
            self.log_message("Dual recording stopped")
            
    def update_recording_timer(self):
        """Update recording countdown timer"""
        if self.recording:
            self.recording_remaining -= 1
            
            if self.recording_remaining <= 0:
                self.stop_recording()
                self.log_message("Dual recording completed (60 seconds)")
            else:
                self.record_btn.setText(f"{self.recording_remaining}s")
                
    def save_data_to_csv(self, timestamp, ch0_raw, ch0_filtered, ch1_raw, ch1_filtered):
        """Save dual data point to CSV if recording"""
        if self.recording and self.csv_writer:
            try:
                self.csv_writer.writerow([f"{timestamp:.6f}", ch0_raw, f"{ch0_filtered:.3f}", 
                                        ch1_raw, f"{ch1_filtered:.3f}"])
                self.csv_file.flush()
            except Exception as e:
                self.log_message(f"CSV write error: {str(e)}")

    def refresh_ports(self):
        """Refresh available serial ports"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")
            
    def toggle_connection(self):
        """Toggle serial connection"""
        if not self.is_connected:
            # Connect
            if self.port_combo.currentText():
                port_name = self.port_combo.currentText().split(' - ')[0]
                baudrate = int(self.baudrate_combo.currentText())
                
                if self.serial_worker.connect_serial(port_name, baudrate):
                    self.serial_worker.start_reading()
                    self.is_connected = True
                    self.connect_btn.setText("Disconnect")
                    self.status_label.setText("Connected")
                    self.status_label.setStyleSheet("color: green; font-weight: bold;")
                    
                    # Start plot update timer
                    self.update_timer.start(50)
                    self.log_message("Dual channel plot timer started")
                    
                    # Reset time to 0 when connecting
                    if hasattr(self, 'serial_worker'):
                        self.serial_worker.current_sample_count = 0
                    self.log_message("Time reset to 0 on connect")
                    
                    # Enable recording button
                    self.record_btn.setEnabled(True)
                    
                    # Disable port/baudrate changes while connected
                    self.port_combo.setEnabled(False)
                    self.baudrate_combo.setEnabled(False)
                    self.refresh_btn.setEnabled(False)
        else:
            # Disconnect
            self.serial_worker.disconnect_serial()
            self.is_connected = False
            self.connect_btn.setText("Connect")
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            
            # Stop plot update timer
            self.update_timer.stop()
            self.log_message("Dual channel plot timer stopped")
            
            # Stop recording if active
            if self.recording:
                self.stop_recording()
                
            # Disable recording button
            self.record_btn.setEnabled(False)
            
            # Re-enable controls
            self.port_combo.setEnabled(True)
            self.baudrate_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            
    def toggle_log(self):
        """Toggle log area visibility"""
        if self.log_visible:
            # Hide log
            self.log_text.hide()
            self.log_btn.setText("Show Log")
            self.log_visible = False
            # Adjust splitter to give all space to plots
            self.splitter.setSizes([400, 400, 0])
        else:
            # Show log
            self.log_text.show()
            self.log_btn.setText("Hide Log")
            self.log_visible = True
            # Restore splitter proportions
            self.splitter.setSizes([300, 300, 150])
        
    def log_message(self, message):
        """Add message to log text area"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
    def process_data(self, data_packet):
        """Process incoming dual channel data from serial worker"""
        timestamps, ch0_values, ch1_values = data_packet
        
        # Set start time on first data (always 0)
        if self.start_time is None:
            self.start_time = 0
            self.log_message("Dual channel start time set to 0")
            
        # Process each data point pair
        for timestamp, ch0_raw, ch1_raw in zip(timestamps, ch0_values, ch1_values):
            # Store raw data
            self.time_data.append(timestamp)
            self.ch0_raw_data.append(ch0_raw)
            self.ch1_raw_data.append(ch1_raw)
            
            # Process through LMS filters
            if self.filter_enabled_ch0:
                ch0_filtered = self.lms_filter_ch0.process_sample(ch0_raw)
            else:
                ch0_filtered = ch0_raw
                
            if self.filter_enabled_ch1:
                ch1_filtered = self.lms_filter_ch1.process_sample(ch1_raw)
            else:
                ch1_filtered = ch1_raw
                
            self.ch0_filtered_data.append(ch0_filtered)
            self.ch1_filtered_data.append(ch1_filtered)
            
            # Save to CSV if recording
            self.save_data_to_csv(timestamp, ch0_raw, ch0_filtered, ch1_raw, ch1_filtered)
            
        # Debug: Log detailed info
        if len(ch0_values) > 0:
            first_time = timestamps[0]
            last_time = timestamps[-1]
            self.log_message(f"Dual Data: {len(ch0_values)} pairs, Time: {first_time:.3f}s to {last_time:.3f}s")
            self.log_message(f"CH0: {min(ch0_values)}-{max(ch0_values)}, CH1: {min(ch1_values)}-{max(ch1_values)}")
            self.log_message(f"Buffer size: {len(self.time_data)} samples")
        
    def process_info(self, info):
        """Process performance info from ESP32"""
        self.current_performance = info
        
        # Update performance labels
        self.sps_label.setText(f"SPS: {info.get('sps', '--')}")
        self.ch0_sps_label.setText(f"CH0: {info.get('ch0_sps', '--')}")
        self.ch1_sps_label.setText(f"CH1: {info.get('ch1_sps', '--')}")
        self.efficiency_label.setText(f"Efficiency: {info.get('efficiency', '--')}%")
        self.skipped_label.setText(f"Skipped: {info.get('skipped', '--')}")
        self.total_label.setText(f"Total: {info.get('total', '--')}")
        
        # Log performance info
        self.log_message(f"Performance - SPS: {info.get('sps', 0)}, CH0: {info.get('ch0_sps', 0)}, CH1: {info.get('ch1_sps', 0)}, Efficiency: {info.get('efficiency', 0)}%")
        
    def process_warning(self, warning):
        """Process warning message from ESP32"""
        self.log_message(f"⚠️ WARNING: {warning}")
        # Also show in status bar temporarily
        self.status_bar.showMessage(f"Warning: {warning}", 5000)
        
    def update_plot(self):
        """Update plots with current dual channel data"""
        if len(self.time_data) > 0:
            # Convert to numpy arrays for plotting
            time_array = np.array(self.time_data)
            ch0_raw_array = np.array(self.ch0_raw_data)
            ch0_filtered_array = np.array(self.ch0_filtered_data)
            ch1_raw_array = np.array(self.ch1_raw_data)
            ch1_filtered_array = np.array(self.ch1_filtered_data)
            
            # Debug: Log plot data info
            if len(time_array) > 0:
                time_range = f"{time_array[0]:.3f}s to {time_array[-1]:.3f}s"
                ch0_range = f"{ch0_raw_array.min()} to {ch0_raw_array.max()}"
                ch1_range = f"{ch1_raw_array.min()} to {ch1_raw_array.max()}"
                
                # Only log every 20 updates to avoid spam
                if hasattr(self, 'plot_update_count'):
                    self.plot_update_count += 1
                else:
                    self.plot_update_count = 1
                    
                if self.plot_update_count % 20 == 0:
                    self.log_message(f"Dual Plot update #{self.plot_update_count}: {len(time_array)} points")
                    self.log_message(f"Time: {time_range}, CH0: {ch0_range}, CH1: {ch1_range}")
            
            # Update raw plots
            self.plot_line_ch0_raw.setData(time_array, ch0_raw_array)
            self.plot_line_ch1_raw.setData(time_array, ch1_raw_array)
            
            # Update filtered plots
            self.plot_line_ch0_filtered.setData(time_array, ch0_filtered_array)
            self.plot_line_ch1_filtered.setData(time_array, ch1_filtered_array)
            
            # Update x-axis range to show last 10 seconds for both plots
            if len(time_array) > 0:
                current_time = time_array[-1]
                x_min = current_time - self.window_seconds
                x_max = current_time
                self.plot_widget_raw.setXRange(x_min, x_max)
                self.plot_widget_filtered.setXRange(x_min, x_max)
                
            # Update status bar
            efficiency = self.current_performance.get('efficiency', 0)
            sps = self.current_performance.get('sps', 0)
            ch0_sps = self.current_performance.get('ch0_sps', 0)
            ch1_sps = self.current_performance.get('ch1_sps', 0)
            ch0_filter_status = "ON" if self.filter_enabled_ch0 else "OFF"
            ch1_filter_status = "ON" if self.filter_enabled_ch1 else "OFF"
            
            self.status_bar.showMessage(
                f"Samples: {len(self.ch0_raw_data)} | SPS: {sps} (CH0:{ch0_sps}, CH1:{ch1_sps}) | "
                f"Efficiency: {efficiency}% | Filter: CH0:{ch0_filter_status}, CH1:{ch1_filter_status} | "
                f"Buffer: {len(self.ch0_raw_data)}/{self.max_samples} | Last: {time_array[-1]:.2f}s"
            )
            
    def update_status(self, message):
        """Update status from serial worker"""
        self.log_message(f"Status: {message}")
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.recording:
            self.stop_recording()
        if self.is_connected:
            self.serial_worker.disconnect_serial()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = DualEKGMonitor()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()