"""
File: websocket_ekg_monitor.py
Version: 2.0
Description: WebSocket server dengan GUI monitoring untuk EKG data dari ESP32 ADS1115
             Compatible dengan format [timestamp,hexdata] batching
"""

import sys
import time
import threading
import numpy as np
import asyncio
import websockets
import re
import os
import csv
from collections import deque
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                           QHBoxLayout, QWidget, QComboBox, QPushButton, 
                           QLabel, QStatusBar, QGroupBox, QTextEdit, QSplitter,
                           QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt, QThread
from PyQt5.QtGui import QFont, QTextCursor
import pyqtgraph as pg

class LMSFilter:
    """Adaptive LMS Predictor Filter for EKG signal processing"""
    
    def __init__(self, filter_order=20, step_size=0.001):
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

class WebSocketWorker(QObject):
    """WebSocket server worker thread"""
    data_received = pyqtSignal(list)
    info_received = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    client_connected = pyqtSignal(str)
    client_disconnected = pyqtSignal(str)
    
    def __init__(self, host="192.168.139.199", port=8765):
        super().__init__()
        self.host = host
        self.port = port
        self.is_running = False
        self.clients = set()
        self.server = None
        self.loop = None
        self.total_messages = 0
        self.total_batches = 0
        self.start_time = time.time()
        self.message_pattern = re.compile(r'\[(\d+),([0-9A-F]+)\]')
        
    def start_server(self):
        """Start WebSocket server with proper cleanup"""
        self.is_running = True
        
        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._run_server())
        except Exception as e:
            self.status_changed.emit(f"Server error: {str(e)}")
        finally:
            # Proper cleanup
            self._cleanup_server()
    
    def stop_server(self):
        """Stop WebSocket server with proper cleanup"""
        self.is_running = False
        
        if self.loop and self.loop.is_running():
            # Schedule cleanup in the loop
            asyncio.run_coroutine_threadsafe(self._async_cleanup(), self.loop)
            
            # Give some time for cleanup
            time.sleep(0.5)
            
            # Stop the loop
            self.loop.call_soon_threadsafe(self.loop.stop)
    
    async def _async_cleanup(self):
        """Async cleanup of server resources"""
        try:
            # Close all client connections
            if self.clients:
                close_tasks = []
                for client in list(self.clients):
                    if not client.closed:
                        close_tasks.append(client.close())
                
                if close_tasks:
                    await asyncio.gather(*close_tasks, return_exceptions=True)
                    await asyncio.sleep(0.1)  # Give time for connections to close
            
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.server = None
                
            self.clients.clear()
            self.status_changed.emit("Server stopped and cleaned up")
            
        except Exception as e:
            self.status_changed.emit(f"Cleanup error: {str(e)}")
    
    def _cleanup_server(self):
        """Final cleanup of event loop"""
        if self.loop:
            try:
                # Close any remaining tasks
                pending = asyncio.all_tasks(self.loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    
                    # Wait for tasks to be cancelled
                    self.loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                
                # Close the loop
                self.loop.close()
                self.loop = None
                
            except Exception as e:
                self.status_changed.emit(f"Loop cleanup error: {str(e)}")
    
    async def _run_server(self):
        """Run WebSocket server with proper socket reuse"""
        # Server settings tanpa reuse_port
        server_settings = {
            'compression': None,
            'max_size': 2**20,
            'max_queue': 100,
            'ping_interval': 20,
            'ping_timeout': 10,
        }
        
        self.status_changed.emit(f"Starting WebSocket server on {self.host}:{self.port}")
        
        try:
            # Method 1: Try standard websockets.serve first
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                **server_settings
            )
            
            self.status_changed.emit("WebSocket server started, waiting for ESP32...")
            
            # Wait until server is stopped
            await self.server.wait_closed()
            
        except OSError as e:
            if "Address already in use" in str(e) or "Only one usage" in str(e):
                self.status_changed.emit(f"Port {self.port} is busy, trying alternative method...")
                
                # Method 2: Manual socket creation with SO_REUSEADDR
                await self._create_server_with_socket_reuse(server_settings)
                
            else:
                raise
    
    async def _create_server_with_socket_reuse(self, server_settings):
        """Create server with manual socket configuration"""
        import socket
        
        # Create socket with reuse address
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            # Set socket options for reuse
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Try to set SO_REUSEPORT if available (Linux/macOS)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                # SO_REUSEPORT not available on Windows
                pass
            
            # Set socket to non-blocking
            sock.setblocking(False)
            
            # Bind and listen
            sock.bind((self.host, self.port))
            sock.listen(5)
            
            # Create server with existing socket
            self.server = await websockets.serve(
                self.handle_client,
                sock=sock,
                **server_settings
            )
            
            self.status_changed.emit("WebSocket server started (with socket reuse)")
            await self.server.wait_closed()
            
        except Exception as sock_error:
            self.status_changed.emit(f"Socket creation failed: {str(sock_error)}")
            
            # Method 3: Wait and retry
            await self._wait_and_retry(server_settings)
            
        finally:
            try:
                sock.close()
            except:
                pass
    
    async def _wait_and_retry(self, server_settings):
        """Wait for socket to be released and retry"""
        self.status_changed.emit("Waiting for socket to be released...")
        
        for attempt in range(1, 6):  # Try 5 times
            await asyncio.sleep(2)  # Wait 2 seconds
            
            try:
                self.status_changed.emit(f"Retry attempt {attempt}/5...")
                
                self.server = await websockets.serve(
                    self.handle_client,
                    self.host,
                    self.port,
                    **server_settings
                )
                
                self.status_changed.emit(f"WebSocket server started (attempt {attempt})")
                await self.server.wait_closed()
                return
                
            except OSError as retry_error:
                if attempt == 5:  # Last attempt
                    self.status_changed.emit(f"All retry attempts failed: {str(retry_error)}")
                    self.status_changed.emit("Try changing the port number or wait longer before restart")
                    raise
                else:
                    self.status_changed.emit(f"Attempt {attempt} failed, retrying...")
                    continue
    
    async def handle_client(self, websocket):
        """Handle client connection with proper cleanup"""
        client_ip = websocket.remote_address[0]
        self.clients.add(websocket)
        self.client_connected.emit(client_ip)
        
        try:
            async for message in websocket:
                if not self.is_running:
                    break
                await self.process_batch_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.client_disconnected.emit(client_ip)
        except Exception as e:
            self.status_changed.emit(f"Client error: {str(e)}")
        finally:
            self.clients.discard(websocket)
            if not websocket.closed:
                try:
                    await websocket.close()
                except:
                    pass
    
    async def process_batch_message(self, websocket, batch_message):
        """Process batched message from ESP32"""
        try:
            self.total_batches += 1
            
            # Parse multiple messages in batch
            messages = self.parse_batch(batch_message)
            
            if messages:
                self.total_messages += len(messages)
                
                # Emit each message separately
                for timestamp, hex_data in messages:
                    decimal_value = int(hex_data, 16)
                    # Convert ESP32 microseconds to seconds
                    timestamp_seconds = timestamp / 1000000.0
                    self.data_received.emit([timestamp_seconds, decimal_value])
                
                # Send ACK for every 10 batches
                if self.total_batches % 10 == 0:
                    ack_message = f"[ACK_BATCH,{self.total_batches},{len(messages)}]"
                    try:
                        await websocket.send(ack_message)
                    except:
                        pass
                        
                # Emit performance info
                if self.total_batches % 50 == 0:  # Every 50 batches
                    elapsed = time.time() - self.start_time
                    sps = self.total_messages / elapsed if elapsed > 0 else 0
                    info = {
                        'sps': int(sps),
                        'efficiency': min(100, int(sps * 100 / 860)),
                        'total': self.total_messages,
                        'batches': self.total_batches
                    }
                    self.info_received.emit(info)
                        
        except Exception as e:
            self.status_changed.emit(f"Error processing batch: {str(e)}")
    
    def parse_batch(self, batch_message):
        """Parse batched message format: [ts1,hex1][ts2,hex2]..."""
        try:
            matches = self.message_pattern.findall(batch_message)
            
            parsed_messages = []
            for timestamp_str, hex_data in matches:
                try:
                    timestamp = int(timestamp_str)
                    parsed_messages.append((timestamp, hex_data))
                except ValueError:
                    continue
                    
            return parsed_messages
            
        except Exception as e:
            self.status_changed.emit(f"Parse error: {str(e)}")
            return []

class WebSocketEKGMonitor(QMainWindow):
    """Main WebSocket EKG Monitor Application"""
    
    def __init__(self):
        super().__init__()
        self.setupLMSFilter()
        self.setupUI()
        self.setupWebSocket()
        self.setupData()
        self.setupTimer()
        self.setupRecording()
        
    def setupLMSFilter(self):
        """Setup LMS filter"""
        self.lms_filter = LMSFilter(filter_order=20, step_size=0.001)
        self.filter_enabled = True
        
    def setupRecording(self):
        """Setup CSV recording functionality"""
        self.recording = False
        self.recording_start_time = None
        self.recording_duration = 60
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
        self.setWindowTitle("WebSocket EKG Monitor with LMS Filter v2.0")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header control panel
        control_group = QGroupBox("WebSocket Server Control")
        control_layout = QHBoxLayout(control_group)
        
        # Server IP input
        control_layout.addWidget(QLabel("Server IP:"))
        self.ip_input = QLineEdit("192.168.139.199")
        self.ip_input.setMinimumWidth(150)
        control_layout.addWidget(self.ip_input)
        
        # Port selection with auto-increment
        control_layout.addWidget(QLabel("Port:"))
        self.port_input = QLineEdit("8765")
        self.port_input.setMinimumWidth(80)
        control_layout.addWidget(self.port_input)
        
        # Auto-find port button
        self.find_port_btn = QPushButton("Find Port")
        self.find_port_btn.clicked.connect(self.find_available_port)
        control_layout.addWidget(self.find_port_btn)
        
        # Start/Stop server button
        self.server_btn = QPushButton("Start Server")
        self.server_btn.clicked.connect(self.toggle_server)
        control_layout.addWidget(self.server_btn)
        
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
        
        # Server status
        self.status_label = QLabel("Server Stopped")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_group)
        
        # Filter Control Panel
        filter_group = QGroupBox("LMS Filter Control")
        filter_layout = QHBoxLayout(filter_group)
        
        # Enable filter checkbox
        self.filter_enable_cb = QCheckBox("Enable Filter")
        self.filter_enable_cb.setChecked(True)
        self.filter_enable_cb.stateChanged.connect(self.toggle_filter)
        filter_layout.addWidget(self.filter_enable_cb)
        
        # Filter order control
        filter_layout.addWidget(QLabel("Order:"))
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(5, 50)
        self.filter_order_spin.setValue(20)
        self.filter_order_spin.valueChanged.connect(self.update_filter_params)
        filter_layout.addWidget(self.filter_order_spin)
        
        # Step size control
        filter_layout.addWidget(QLabel("Step Size (μ):"))
        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(0.001, 0.1)
        self.step_size_spin.setValue(0.001)
        self.step_size_spin.setDecimals(3)
        self.step_size_spin.setSingleStep(0.001)
        self.step_size_spin.valueChanged.connect(self.update_filter_params)
        filter_layout.addWidget(self.step_size_spin)
        
        # Reset filter button
        self.reset_filter_btn = QPushButton("Reset Filter")
        self.reset_filter_btn.clicked.connect(self.reset_filter)
        filter_layout.addWidget(self.reset_filter_btn)
        
        # Show/Hide filtered plot button
        self.show_filtered_btn = QPushButton("Hide Filtered")
        self.show_filtered_btn.clicked.connect(self.toggle_filtered_plot)
        filter_layout.addWidget(self.show_filtered_btn)
        
        filter_layout.addStretch()
        main_layout.addWidget(filter_group)
        
        # Performance metrics panel
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QHBoxLayout(metrics_group)
        
        self.sps_label = QLabel("SPS: --")
        self.efficiency_label = QLabel("Efficiency: --%")
        self.total_label = QLabel("Total: --")
        self.clients_label = QLabel("Clients: 0")
        
        metrics_layout.addWidget(self.sps_label)
        metrics_layout.addWidget(self.efficiency_label)
        metrics_layout.addWidget(self.total_label)
        metrics_layout.addWidget(self.clients_label)
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
        self.plot_line_raw = self.plot_widget_raw.plot(pen=pg.mkPen(color='red', width=2))
        splitter.addWidget(self.plot_widget_raw)
        
        # Filtered signal plot widget
        self.plot_widget_filtered = pg.PlotWidget()
        self.plot_widget_filtered.setLabel('left', 'Amplitude', units='')
        self.plot_widget_filtered.setLabel('bottom', 'Time', units='s')
        self.plot_widget_filtered.setTitle('EKG Signal - LMS Filtered (10 second window)')
        self.plot_widget_filtered.setYRange(0, 4095)
        self.plot_widget_filtered.showGrid(x=True, y=True)
        self.plot_widget_filtered.setBackground('white')
        self.plot_line_filtered = self.plot_widget_filtered.plot(pen=pg.mkPen(color='blue', width=2))
        splitter.addWidget(self.plot_widget_filtered)
        self.filtered_plot_visible = True
        
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
        self.status_bar.showMessage("Ready - WebSocket Mode")
        
    def setupWebSocket(self):
        """Setup WebSocket worker"""
        self.websocket_worker = None
        self.websocket_thread = None
        self.server_running = False
        self.connected_clients = 0
        
    def setupData(self):
        """Setup data storage"""
        self.window_seconds = 10
        self.max_samples = 8600  # 860 SPS * 10 seconds
        self.time_data = deque(maxlen=self.max_samples)
        self.signal_data_raw = deque(maxlen=self.max_samples)
        self.signal_data_filtered = deque(maxlen=self.max_samples)
        self.current_performance = {}
        
    def setupTimer(self):
        """Setup update timer"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        
    def find_available_port(self):
        """Find available port starting from current port"""
        import socket
        
        start_port = int(self.port_input.text())
        
        for port in range(start_port, start_port + 10):
            try:
                # Test if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((self.ip_input.text(), port))
                    
                # Port is available
                self.port_input.setText(str(port))
                self.log_message(f"Found available port: {port}")
                return
                
            except OSError:
                continue
        
        self.log_message(f"No available ports found in range {start_port}-{start_port+9}")
        
    def toggle_server(self):
        """Toggle WebSocket server with port validation"""
        if not self.server_running:
            # Check if port is available before starting
            if self.is_port_available():
                self.start_server()
            else:
                self.log_message("Port is busy, trying to find alternative...")
                self.find_available_port()
        else:
            self.stop_server()
    
    def is_port_available(self):
        """Check if port is available"""
        import socket
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.ip_input.text(), int(self.port_input.text())))
                return True
        except OSError:
            return False
            
    def start_server(self):
        """Start WebSocket server"""
        host = self.ip_input.text()
        port = int(self.port_input.text())
        
        # Create worker and thread
        self.websocket_worker = WebSocketWorker(host, port)
        self.websocket_thread = QThread()
        
        # Move worker to thread
        self.websocket_worker.moveToThread(self.websocket_thread)
        
        # Connect signals
        self.websocket_worker.data_received.connect(self.process_data)
        self.websocket_worker.info_received.connect(self.process_info)
        self.websocket_worker.status_changed.connect(self.update_status)
        self.websocket_worker.client_connected.connect(self.client_connected)
        self.websocket_worker.client_disconnected.connect(self.client_disconnected)
        
        # Connect thread signals
        self.websocket_thread.started.connect(self.websocket_worker.start_server)
        
        # Start thread
        self.websocket_thread.start()
        
        # Update UI
        self.server_running = True
        self.server_btn.setText("Stop Server")
        self.status_label.setText("Server Running")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        # Disable server config
        self.ip_input.setEnabled(False)
        self.port_input.setEnabled(False)
        self.find_port_btn.setEnabled(False)
        
        # Start plot update timer
        self.update_timer.start(50)
        
        # Enable recording
        self.record_btn.setEnabled(True)
        
        self.log_message(f"WebSocket server started on {host}:{port}")
        
    def stop_server(self):
        """Stop WebSocket server with proper cleanup"""
        if self.websocket_worker:
            self.websocket_worker.stop_server()
            
        if self.websocket_thread:
            # Give more time for cleanup
            self.websocket_thread.quit()
            if not self.websocket_thread.wait(5000):  # Wait up to 5 seconds
                self.log_message("Warning: Thread didn't finish cleanly, terminating...")
                self.websocket_thread.terminate()
                self.websocket_thread.wait()
            
        # Clean up references
        self.websocket_worker = None
        self.websocket_thread = None
        
        # Update UI
        self.server_running = False
        self.server_btn.setText("Start Server")
        self.status_label.setText("Server Stopped")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        # Enable server config
        self.ip_input.setEnabled(True)
        self.port_input.setEnabled(True)
        self.find_port_btn.setEnabled(True)
        
        # Stop timers
        self.update_timer.stop()
        if self.recording:
            self.stop_recording()
            
        # Disable recording
        self.record_btn.setEnabled(False)
        
        # Reset client count
        self.connected_clients = 0
        self.clients_label.setText("Clients: 0")
        
        self.log_message("WebSocket server stopped and cleaned up")
        
        # Force garbage collection
        import gc
        gc.collect()
        
    def client_connected(self, client_ip):
        """Handle client connection"""
        self.connected_clients += 1
        self.clients_label.setText(f"Clients: {self.connected_clients}")
        self.log_message(f"ESP32 client connected: {client_ip}")
        
    def client_disconnected(self, client_ip):
        """Handle client disconnection"""
        self.connected_clients = max(0, self.connected_clients - 1)
        self.clients_label.setText(f"Clients: {self.connected_clients}")
        self.log_message(f"ESP32 client disconnected: {client_ip}")
        
    def toggle_filter(self, state):
        """Toggle filter enable/disable"""
        self.filter_enabled = state == Qt.Checked
        self.log_message(f"LMS Filter {'enabled' if self.filter_enabled else 'disabled'}")
            
    def update_filter_params(self):
        """Update filter parameters"""
        filter_order = self.filter_order_spin.value()
        step_size = self.step_size_spin.value()
        self.lms_filter.update_parameters(filter_order, step_size)
        self.log_message(f"Filter params updated: Order={filter_order}, μ={step_size}")
        
    def reset_filter(self):
        """Reset filter coefficients"""
        self.lms_filter.reset()
        self.log_message("LMS Filter reset")
        
    def toggle_filtered_plot(self):
        """Toggle filtered plot visibility"""
        if self.filtered_plot_visible:
            self.plot_widget_filtered.hide()
            self.show_filtered_btn.setText("Show Filtered")
            self.filtered_plot_visible = False
            if self.log_visible:
                self.splitter.setSizes([600, 0, 150])
            else:
                self.splitter.setSizes([800, 0, 0])
        else:
            self.plot_widget_filtered.show()
            self.show_filtered_btn.setText("Hide Filtered")
            self.filtered_plot_visible = True
            if self.log_visible:
                self.splitter.setSizes([300, 300, 150])
            else:
                self.splitter.setSizes([400, 400, 0])
                
    def toggle_log(self):
        """Toggle log area visibility"""
        if self.log_visible:
            self.log_text.hide()
            self.log_btn.setText("Show Log")
            self.log_visible = False
            if self.filtered_plot_visible:
                self.splitter.setSizes([400, 400, 0])
            else:
                self.splitter.setSizes([800, 0, 0])
        else:
            self.log_text.show()
            self.log_btn.setText("Hide Log")
            self.log_visible = True
            if self.filtered_plot_visible:
                self.splitter.setSizes([300, 300, 150])
            else:
                self.splitter.setSizes([600, 0, 150])
        
    def reset_signal(self):
        """Reset signal data and plot"""
        self.time_data.clear()
        self.signal_data_raw.clear()
        self.signal_data_filtered.clear()
        self.plot_line_raw.setData([], [])
        self.plot_line_filtered.setData([], [])
        self.log_text.clear()
        self.status_bar.showMessage("Signal reset - WebSocket Mode")
        self.log_message("Signal data reset")
        self.lms_filter.reset()
        
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
            filename = f"log/websocket_{timestamp}.csv"
            
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp_seconds', 'raw_value', 'filtered_value'])
            
            self.recording = True
            self.recording_start_time = time.time()
            self.recording_remaining = self.recording_duration
            
            self.record_btn.setText(f"{self.recording_remaining}s")
            self.record_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
            
            self.recording_timer.start(1000)
            self.log_message(f"Recording started: {filename}")
            
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
            
            self.log_message("Recording stopped")
            
    def update_recording_timer(self):
        """Update recording countdown timer"""
        if self.recording:
            self.recording_remaining -= 1
            
            if self.recording_remaining <= 0:
                self.stop_recording()
                self.log_message("Recording completed (60 seconds)")
            else:
                self.record_btn.setText(f"{self.recording_remaining}s")
                
    def save_data_to_csv(self, timestamp, raw_value, filtered_value):
        """Save data point to CSV if recording"""
        if self.recording and self.csv_writer:
            try:
                self.csv_writer.writerow([f"{timestamp:.6f}", raw_value, f"{filtered_value:.3f}"])
                self.csv_file.flush()
            except Exception as e:
                self.log_message(f"CSV write error: {str(e)}")
                
    def log_message(self, message):
        """Add message to log text area"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
    def process_data(self, data_point):
        """Process incoming data point"""
        timestamp, raw_value = data_point
        
        # Store raw data
        self.time_data.append(timestamp)
        self.signal_data_raw.append(raw_value)
        
        # Process through LMS filter
        if self.filter_enabled:
            filtered_value = self.lms_filter.process_sample(raw_value)
        else:
            filtered_value = raw_value
            
        self.signal_data_filtered.append(filtered_value)
        
        # Save to CSV if recording
        self.save_data_to_csv(timestamp, raw_value, filtered_value)
        
        # Debug log every 1000 samples
        if len(self.time_data) % 1000 == 0:
            self.log_message(f"Data point: T={timestamp:.3f}s, Raw={raw_value}, Filtered={filtered_value:.1f}")
            
    def process_info(self, info):
        """Process performance info"""
        self.current_performance = info
        
        self.sps_label.setText(f"SPS: {info.get('sps', '--')}")
        self.efficiency_label.setText(f"Efficiency: {info.get('efficiency', '--')}%")
        self.total_label.setText(f"Total: {info.get('total', '--')}")
        
        self.log_message(f"Performance - SPS: {info.get('sps', 0)}, Efficiency: {info.get('efficiency', 0)}%")
        
    def update_status(self, message):
        """Update status from WebSocket worker"""
        self.log_message(f"Status: {message}")
        
    def update_plot(self):
        """Update plots with current data"""
        if len(self.time_data) > 0:
            time_array = np.array(self.time_data)
            raw_array = np.array(self.signal_data_raw)
            filtered_array = np.array(self.signal_data_filtered)
            
            # Update plots
            self.plot_line_raw.setData(time_array, raw_array)
            self.plot_line_filtered.setData(time_array, filtered_array)
            
            # Update x-axis range to show last 10 seconds
            if len(time_array) > 0:
                current_time = time_array[-1]
                x_min = current_time - self.window_seconds
                x_max = current_time + 0.5
                self.plot_widget_raw.setXRange(x_min, x_max)
                self.plot_widget_filtered.setXRange(x_min, x_max)
                
            # Update status bar
            efficiency = self.current_performance.get('efficiency', 0)
            sps = self.current_performance.get('sps', 0)
            filter_status = "ON" if self.filter_enabled else "OFF"
            self.status_bar.showMessage(
                f"Samples: {len(self.signal_data_raw)} | SPS: {sps} | Efficiency: {efficiency}% | "
                f"Filter: {filter_status} | Time: {time_array[-1]:.2f}s | Mode: WebSocket"
            )
            
    def closeEvent(self, event):
        """Handle application close"""
        if self.recording:
            self.stop_recording()
        if self.server_running:
            self.stop_server()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = WebSocketEKGMonitor()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()