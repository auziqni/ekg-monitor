import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit
)
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtCore import QIODevice

class SerialMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UART Monitor (PyQt5)")

        self.serial = QSerialPort(self)
        self.serial.readyRead.connect(self.read_data)
        self.serial.errorOccurred.connect(self.on_error)

        self.portCombo = QComboBox()
        self.refreshButton = QPushButton("Refresh")

        self.baudCombo = QComboBox()
        self.baudCombo.setEditable(True)
        for b in [9600, 19200, 38400, 57600, 115200,
                  230400, 250000, 256000, 460800, 500000,
                  921600, 1000000, 1500000, 2000000]:
            self.baudCombo.addItem(str(b))
        self.baudCombo.setCurrentText("115200")

        self.connectButton = QPushButton("Connect")
        self.clearButton = QPushButton("Clear")
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        top = QHBoxLayout()
        top.addWidget(QLabel("Port:"))
        top.addWidget(self.portCombo, 1)
        top.addWidget(self.refreshButton)
        top.addWidget(QLabel("Baud:"))
        top.addWidget(self.baudCombo)
        top.addWidget(self.connectButton)
        top.addWidget(self.clearButton)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.output, 1)

        self.refreshButton.clicked.connect(self.populate_ports)
        self.connectButton.clicked.connect(self.toggle_connection)
        self.clearButton.clicked.connect(self.output.clear)

        self.populate_ports()

    def populate_ports(self):
        self.portCombo.clear()
        for p in QSerialPortInfo.availablePorts():
            label = f"{p.portName()} - {p.description()}"
            self.portCombo.addItem(label, p.portName())

    def toggle_connection(self):
        if self.serial.isOpen():
            self.serial.close()
            self.connectButton.setText("Connect")
            self.output.append("[i] Disconnected")
            return

        if self.portCombo.count() == 0:
            self.output.append("[!] No ports available")
            return

        port_name = self.portCombo.currentData()
        try:
            baud = int(self.baudCombo.currentText())
        except ValueError:
            self.output.append("[!] Invalid baud")
            return

        self.serial.setPortName(port_name)
        self.serial.setBaudRate(baud)
        self.serial.setDataBits(QSerialPort.Data8)
        self.serial.setParity(QSerialPort.NoParity)
        self.serial.setStopBits(QSerialPort.OneStop)
        self.serial.setFlowControl(QSerialPort.NoFlowControl)

        if self.serial.open(QIODevice.ReadOnly):
            self.connectButton.setText("Disconnect")
            self.output.append(f"[i] Connected to {port_name} @ {baud}")
        else:
            self.output.append(f"[!] Failed to open {port_name}: {self.serial.errorString()}")

    def read_data(self):
        data = self.serial.readAll()
        text = bytes(data).decode("utf-8", errors="replace")
        self.output.moveCursor(self.output.textCursor().End)
        self.output.insertPlainText(text)
        self.output.moveCursor(self.output.textCursor().End)

    def on_error(self, err):
        if err == QSerialPort.NoError or not self.serial.isOpen():
            return
        self.output.append(f"[err] {self.serial.errorString()}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SerialMonitor()
    w.resize(900, 500)
    w.show()
    sys.exit(app.exec_())