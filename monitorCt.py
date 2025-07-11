# filename: monitorCt.py
# revisi: v2.0  
# short description: Real-time continuous EKG signal monitoring with hex CSV format,
#                   auto-save CSV recording (60s max), manual start/stop controls,
#                   and 10-second sliding window display for ESP32 + ADS1115

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
                           QLabel, QStatusBar, QGroupBox, QTextEdit, QSplitter)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont, QTextCursor
import pyqtgraph as pg

class SerialWorkerContinuous(QObject):
    """Worker thread for continuous hex CSV serial communication"""
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
        self.current_sample_count = 0  # Track total samples received
        
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
        self.current_sample_count = 0  # Reset sample counter to start from 0
        self.start_time = None
        threading.Thread(target=self._read_loop, daemon=True).start()
    
    def parse_info_line(self, line):
        """Parse info line: # TS:1000000 SPS:860 EFF:98% SKIP:0 TOT:5487"""
        info = {}
        try:
            parts = line[1:].strip().split()  # Remove # and split
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    if key == 'TS':
                        info['timestamp'] = int(value)
                        self.last_timestamp = info['timestamp']
                    elif key == 'SPS':
                        info['sps'] = int(value)
                    elif key == 'EFF':
                        info['efficiency'] = int(value.replace('%', ''))
                    elif key == 'SKIP':
                        info['skipped'] = int(value)
                    elif key == 'TOT':
                        info['total'] = int(value)
        except Exception as e:
            self.status_changed.emit(f"Info parse error: {str(e)}")
        return info
    
    def parse_hex_csv(self, line):
        """Parse hex CSV line: 3FF,415,42B,441,456,46A,47C,48D,49C,4AA"""
        try:
            hex_values = line.strip().split(',')
            decimal_values = [int(hex_val, 16) for hex_val in hex_values if hex_val.strip()]
            return decimal_values
        except Exception as e:
            self.status_changed.emit(f"CSV parse error: {str(e)}")
            return []
    
    def interpolate_timestamps(self, data_count):
        """Create interpolated timestamps for data points starting from 0"""
        timestamps = []
        for i in range(data_count):
            # Calculate time in seconds from start (time = 0)
            time_seconds = (self.current_sample_count + i) * self.sample_interval_us / 1000000.0
            timestamps.append(time_seconds)
            
        # Update sample count for next batch
        self.current_sample_count += data_count
        
        return timestamps
    
    def _read_loop(self):
        """Main reading loop for continuous format"""
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
                    warning_msg = line[1:].strip()  # Remove * prefix
                    self.warning_received.emit(warning_msg)
                    
                else:
                    # Data line (hex CSV)
                    decimal_values = self.parse_hex_csv(line)
                    if decimal_values:
                        # Create timestamps for each value
                        timestamps = self.interpolate_timestamps(len(decimal_values))
                        # Emit data with timestamps
                        self.data_received.emit([timestamps, decimal_values])
                        
            except Exception as e:
                if self.is_running:  # Only report error if still supposed to be running
                    self.status_changed.emit(f"Read error: {str(e)}")
                break

class EKGMonitorContinuous(QMainWindow):
    """Main application window for continuous monitoring"""
    
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.setupSerial()
        self.setupData()
        self.setupTimer()
        self.setupRecording()
        
    def setupRecording(self):
        """Setup CSV recording functionality"""
        self.recording = False
        self.recording_start_time = None
        self.recording_duration = 60  # 60 seconds max
        self.recording_remaining = 0
        self.csv_file = None
        self.csv_writer = None
        
        # Recording countdown timer
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_timer)
        
        # Create log directory if it doesn't exist
        if not os.path.exists('log'):
            os.makedirs('log')
        
    def setupUI(self):
        """Setup user interface"""
        self.setWindowTitle("EKG Continuous Monitor v2.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget with splitter
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
        self.record_btn.setEnabled(False)  # Disabled until connected
        control_layout.addWidget(self.record_btn)
        
        # Connection status
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_group)
        
        # Performance metrics panel
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QHBoxLayout(metrics_group)
        
        self.sps_label = QLabel("SPS: --")
        self.efficiency_label = QLabel("Efficiency: --%")
        self.skipped_label = QLabel("Skipped: --")
        self.total_label = QLabel("Total: --")
        
        metrics_layout.addWidget(self.sps_label)
        metrics_layout.addWidget(self.efficiency_label)
        metrics_layout.addWidget(self.skipped_label)
        metrics_layout.addWidget(self.total_label)
        metrics_layout.addStretch()
        
        main_layout.addWidget(metrics_group)
        
        # Splitter for plot and log
        splitter = QSplitter(Qt.Vertical)
        self.splitter = splitter  # Store reference for show/hide functionality
        
        # Plot widget with white background
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude', units='')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setTitle('EKG Signal - Continuous Real-time (10 second window)')
        self.plot_widget.setYRange(0, 4095)
        self.plot_widget.showGrid(x=True, y=True)
        
        # Set white background
        self.plot_widget.setBackground('white')
        
        # Create plot line
        self.plot_line = self.plot_widget.plot(pen=pg.mkPen(color='red', width=2))
        
        splitter.addWidget(self.plot_widget)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background-color: #f0f0f0;")
        splitter.addWidget(self.log_text)
        self.log_visible = True  # Track log visibility
        
        # Set splitter proportions
        splitter.setSizes([600, 150])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initial port refresh
    def reset_signal(self):
        """Reset signal data and plot"""
        self.time_data.clear()
        self.signal_data.clear()
        self.start_time = None
        self.plot_line.setData([], [])
        self.log_text.clear()
        self.status_bar.showMessage("Signal reset")
        self.log_message("Signal data reset")
        
        # Initial port refresh
        self.refresh_ports()
        
    def setupSerial(self):
        """Setup serial worker"""
        self.serial_worker = SerialWorkerContinuous()
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
        self.signal_data = deque(maxlen=self.max_samples)
        self.start_time = None
        self.current_performance = {}
        
    def setupTimer(self):
        """Setup update timer"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        # Don't start timer here - start when connected
        
    def toggle_recording(self):
        """Toggle CSV recording"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start CSV recording"""
        try:
            # Create filename with current timestamp
            timestamp = time.strftime("%H-%M-%S")
            filename = f"log/{timestamp}.csv"
            
            # Open CSV file for writing
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            self.csv_writer.writerow(['timestamp', 'value'])
            
            # Start recording
            self.recording = True
            self.recording_start_time = time.time()
            self.recording_remaining = self.recording_duration
            
            # Update button appearance
            self.record_btn.setText(f"{self.recording_remaining}s")
            self.record_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
            
            # Start countdown timer
            self.recording_timer.start(1000)  # Update every second
            
            self.log_message(f"Recording started: {filename}")
            
        except Exception as e:
            self.log_message(f"Recording start failed: {str(e)}")
            
    def stop_recording(self):
        """Stop CSV recording"""
        if self.recording:
            self.recording = False
            self.recording_timer.stop()
            
            # Close CSV file
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
            
            # Reset button appearance
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
            
            self.log_message("Recording stopped")
            
    def update_recording_timer(self):
        """Update recording countdown timer"""
        if self.recording:
            self.recording_remaining -= 1
            
            if self.recording_remaining <= 0:
                # Recording time finished
                self.stop_recording()
                self.log_message("Recording completed (60 seconds)")
            else:
                # Update button with remaining time
                self.record_btn.setText(f"{self.recording_remaining}s")
                
    def save_data_to_csv(self, timestamp, value):
        """Save data point to CSV if recording"""
        if self.recording and self.csv_writer:
            try:
                self.csv_writer.writerow([f"{timestamp:.6f}", value])
                self.csv_file.flush()  # Ensure data is written immediately
            except Exception as e:
                self.log_message(f"CSV write error: {str(e)}")

    def refresh_ports(self):
        """Refresh available serial ports"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")
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
                    self.update_timer.start(50)  # Update every 50ms
                    self.log_message("Plot timer started")
                    
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
            self.log_message("Plot timer stopped")
            
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
            # Adjust splitter to give all space to plot
            self.splitter.setSizes([800, 0])
        else:
            # Show log
            self.log_text.show()
            self.log_btn.setText("Hide Log")
            self.log_visible = True
            # Restore splitter proportions
            self.splitter.setSizes([600, 150])
        
    def log_message(self, message):
        """Add message to log text area"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
    def process_data(self, data_packet):
        """Process incoming data from serial worker"""
        timestamps, values = data_packet
        
        # Set start time on first data (always 0)
        if self.start_time is None:
            self.start_time = 0
            self.log_message("Start time set to 0")
            
        # Process each data point
        for timestamp, value in zip(timestamps, values):
            self.time_data.append(timestamp)  # timestamp is already in seconds from 0
            self.signal_data.append(value)
            
            # Save to CSV if recording
            self.save_data_to_csv(timestamp, value)
            
        # Debug: Log detailed info
        if len(values) > 0:
            first_time = timestamps[0]
            last_time = timestamps[-1]
            self.log_message(f"Data: {len(values)} samples, Time: {first_time:.3f}s to {last_time:.3f}s, Values: {min(values)}-{max(values)}")
            self.log_message(f"Hex: [{', '.join(f'{v:03X}' for v in values[:5])}{'...' if len(values) > 5 else ''}]")
            self.log_message(f"Buffer size: {len(self.time_data)} samples")
        
    def process_info(self, info):
        """Process performance info from ESP32"""
        self.current_performance = info
        
        # Update performance labels
        self.sps_label.setText(f"SPS: {info.get('sps', '--')}")
        self.efficiency_label.setText(f"Efficiency: {info.get('efficiency', '--')}%")
        self.skipped_label.setText(f"Skipped: {info.get('skipped', '--')}")
        self.total_label.setText(f"Total: {info.get('total', '--')}")
        
        # Log performance info
        self.log_message(f"Performance - SPS: {info.get('sps', 0)}, Efficiency: {info.get('efficiency', 0)}%, Skipped: {info.get('skipped', 0)}")
        
    def process_warning(self, warning):
        """Process warning message from ESP32"""
        self.log_message(f"⚠️ WARNING: {warning}")
        # Also show in status bar temporarily
        self.status_bar.showMessage(f"Warning: {warning}", 5000)  # 5 seconds
        
    def update_plot(self):
        """Update plot with current data"""
        if len(self.time_data) > 0:
            # Convert to numpy arrays for plotting
            time_array = np.array(self.time_data)
            signal_array = np.array(self.signal_data)
            
            # Debug: Log plot data info
            if len(time_array) > 0:
                time_range = f"{time_array[0]:.3f}s to {time_array[-1]:.3f}s"
                value_range = f"{signal_array.min()} to {signal_array.max()}"
                # Only log every 20 updates to avoid spam
                if hasattr(self, 'plot_update_count'):
                    self.plot_update_count += 1
                else:
                    self.plot_update_count = 1
                    
                if self.plot_update_count % 20 == 0:
                    self.log_message(f"Plot update #{self.plot_update_count}: {len(time_array)} points, Time: {time_range}, Values: {value_range}")
            
            # Update plot
            self.plot_line.setData(time_array, signal_array)
            
            # Update x-axis range to show last 10 seconds
            if len(time_array) > 0:
                current_time = time_array[-1]
                self.plot_widget.setXRange(current_time - self.window_seconds, current_time)
                
            # Update status bar
            efficiency = self.current_performance.get('efficiency', 0)
            sps = self.current_performance.get('sps', 0)
            self.status_bar.showMessage(
                f"Samples: {len(self.signal_data)} | SPS: {sps} | Efficiency: {efficiency}% | "
                f"Buffer: {len(self.signal_data)}/{self.max_samples} | Last: {time_array[-1]:.2f}s"
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
    window = EKGMonitorContinuous()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()