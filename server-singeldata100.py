"""
File: websocket_server.py
Version: 1.1
Description: WebSocket server untuk menerima high-frequency data dari ESP32 (1000 msg/sec)
"""

import asyncio
import websockets
from datetime import datetime
import time

class WebSocketServer:
    def __init__(self, host="192.168.139.199", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.message_count = 0
        self.start_time = time.time()
        self.last_display_time = time.time()
        
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
            self.message_count += 1
            
            # Parse format: [timestamp,hexdata]
            if message.startswith('[') and message.endswith(']'):
                content = message[1:-1]  # Remove brackets
                parts = content.split(',')
                
                if len(parts) == 2:
                    timestamp = parts[0]
                    hex_data = parts[1]
                    
                    # Display every 100th message to avoid spam
                    current_time = time.time()
                    if self.message_count % 100 == 0 or (current_time - self.last_display_time) >= 1.0:
                        try:
                            decimal_value = int(hex_data, 16)
                            elapsed_time = current_time - self.start_time
                            rate = self.message_count / elapsed_time if elapsed_time > 0 else 0
                            
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] #{self.message_count} - "
                                  f"TS:{timestamp}, Hex:{hex_data}, Dec:{decimal_value}, "
                                  f"Rate:{rate:.1f} msg/sec")
                            
                            self.last_display_time = current_time
                        except ValueError:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] #{self.message_count} - "
                                  f"TS:{timestamp}, Hex:{hex_data}")
                    
                    # Send ACK only for every 10th message to reduce traffic
                    if self.message_count % 10 == 0:
                        response = f"[ACK,{self.message_count}]"
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