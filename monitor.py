# filename: monitor.py
# revisi: v1.0
# short description: Real-time EKG signal monitoring application with PyQtGraph,
#                   serial communication, and 10-second sliding window display

import sys
import time
import threading
import numpy as np
import serial
import serial.tools.list_ports
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                           QHBoxLayout, QWidget, QComboBox, QPushButton, 
                           QLabel, QStatusBar, QGroupBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

class SerialWorker(QObject):
    """Worker thread for serial communication"""
    data_received = pyqtSignal(list)
    status_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_running = False
        self.port_name = ""
        self.baudrate = 250000
        
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
        threading.Thread(target=self._read_loop, daemon=True).start()
    
    def _read_loop(self):
        """Main reading loop"""
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                if line and not line.startswith('#'):
                    # Parse data: timestamp,sps,data0,data1,data2...
                    parts = line.split(',')
                    if len(parts) >= 3:
                        timestamp = int(parts[0])
                        sps = int(parts[1])
                        data = [int(x) for x in parts[2:]]
                        self.data_received.emit([timestamp, sps, data])
                elif line.startswith('#'):
                    self.status_changed.emit(line)
            except Exception as e:
                if self.is_running:  # Only report error if still supposed to be running
                    self.status_changed.emit(f"Read error: {str(e)}")
                break

class EKGMonitor(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.setupSerial()
        self.setupData()
        self.setupTimer()
        
    def setupUI(self):
        """Setup user interface"""
        self.setWindowTitle("EKG Signal Monitor v1.0")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
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
        
        # Connection status
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_group)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude', units='')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setTitle('EKG Signal - Real-time (10 second window)')
        self.plot_widget.setYRange(0, 4095)
        self.plot_widget.showGrid(x=True, y=True)
        
        # Create plot line
        self.plot_line = self.plot_widget.plot(pen=pg.mkPen(color='red', width=2))
        
        main_layout.addWidget(self.plot_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initial port refresh
        self.refresh_ports()
        
    def setupSerial(self):
        """Setup serial worker"""
        self.serial_worker = SerialWorker()
        self.serial_worker.data_received.connect(self.process_data)
        self.serial_worker.status_changed.connect(self.update_status)
        self.is_connected = False
        
    def setupData(self):
        """Setup data storage"""
        self.window_seconds = 10
        self.max_samples = 8600  # 860 SPS * 10 seconds
        self.time_data = deque(maxlen=self.max_samples)
        self.signal_data = deque(maxlen=self.max_samples)
        self.start_time = None
        self.current_sps = 0
        self.sample_count = 0
        
    def setupTimer(self):
        """Setup update timer"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(50)  # Update every 50ms for smooth animation
        
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
            
            # Re-enable controls
            self.port_combo.setEnabled(True)
            self.baudrate_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            
    def reset_signal(self):
        """Reset signal data and plot"""
        self.time_data.clear()
        self.signal_data.clear()
        self.start_time = None
        self.sample_count = 0
        self.plot_line.setData([], [])
        self.status_bar.showMessage("Signal reset")
        print("Signal data reset")
        
    def process_data(self, data_packet):
        """Process incoming data from serial worker"""
        timestamp, sps, data_array = data_packet
        self.current_sps = sps
        
        # Set start time on first data
        if self.start_time is None:
            self.start_time = timestamp
            
        # Process each data point
        for i, value in enumerate(data_array):
            # Calculate relative time in seconds
            relative_time = (timestamp - self.start_time) / 1000000.0  # Convert to seconds
            sample_time = relative_time + (i / sps) if sps > 0 else relative_time
            
            self.time_data.append(sample_time)
            self.signal_data.append(value)
            self.sample_count += 1
            
        # Print to terminal
        print(f"Received {len(data_array)} samples, SPS: {sps}, Latest values: {data_array[-5:]}")
        
    def update_plot(self):
        """Update plot with current data"""
        if len(self.time_data) > 0:
            # Convert to numpy arrays for plotting
            time_array = np.array(self.time_data)
            signal_array = np.array(self.signal_data)
            
            # Update plot
            self.plot_line.setData(time_array, signal_array)
            
            # Update x-axis range to show last 10 seconds
            if len(time_array) > 0:
                current_time = time_array[-1]
                self.plot_widget.setXRange(current_time - self.window_seconds, current_time)
                
            # Update status bar
            self.status_bar.showMessage(
                f"Samples: {self.sample_count} | Current SPS: {self.current_sps} | "
                f"Buffer: {len(self.signal_data)}/{self.max_samples}"
            )
            
    def update_status(self, message):
        """Update status from serial worker"""
        if message.startswith('#'):
            # Performance message from ESP32
            print(f"ESP32: {message}")
        else:
            # Status message
            print(f"Status: {message}")
            
    def closeEvent(self, event):
        """Handle application close"""
        if self.is_connected:
            self.serial_worker.disconnect_serial()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = EKGMonitor()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()