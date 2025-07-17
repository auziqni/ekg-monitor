"""
File: websocket_server.py
Version: 1.0
Description: WebSocket server untuk menerima data dari ESP32 dengan format [timestamp,hexdata]
compatible : send-singledata.cpp
"""

import asyncio
import websockets
from datetime import datetime

class WebSocketServer:
    def __init__(self, host="192.168.139.199", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        
    async def register_client(self, websocket):
        """Register new client connection"""
        self.clients.add(websocket)
        client_ip = websocket.remote_address[0]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client connected: {client_ip}")
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Client disconnected: {client_ip}")
        finally:
            self.clients.discard(websocket)
    
    async def handle_message(self, websocket, message):
        """Handle incoming messages from ESP32"""
        try:
            # Parse format: [timestamp,hexdata]
            if message.startswith('[') and message.endswith(']'):
                content = message[1:-1]  # Remove brackets
                parts = content.split(',')
                
                if len(parts) == 2:
                    timestamp = parts[0]
                    hex_data = parts[1]
                    
                    # Convert hex to decimal for display
                    try:
                        decimal_value = int(hex_data, 16)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Received - Timestamp: {timestamp}, Hex: {hex_data}, Decimal: {decimal_value}")
                    except ValueError:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Received - Timestamp: {timestamp}, Hex: {hex_data}")
                    
                    # Echo back to ESP32 (optional)
                    response = f"[ACK,{timestamp}]"
                    await websocket.send(response)
                    
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Invalid format: {message}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Raw message: {message}")
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error handling message: {e}")
    
    async def broadcast_message(self, message):
        """Broadcast message to all connected clients"""
        if self.clients:
            disconnected = []
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.clients.discard(client)
    
    async def start_server(self):
        """Start WebSocket server"""
        print(f"Starting WebSocket server on {self.host}:{self.port}")
        print("Waiting for ESP32 connections...")
        
        async with websockets.serve(
            self.register_client, 
            self.host, 
            self.port
        ):
            await asyncio.Future()  # Run forever

async def main():
    server = WebSocketServer()
    await server.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")