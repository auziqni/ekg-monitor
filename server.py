"""
File: websocket_server.py
Version: 2.0
Description: Optimized WebSocket server untuk handle 860 SPS dengan batching support
"""

import asyncio
import websockets
import time
import re
from datetime import datetime
from collections import deque

class OptimizedWebSocketServer:
    def __init__(self, host="192.168.139.199", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        
        # Performance monitoring
        self.total_messages = 0
        self.total_batches = 0
        self.start_time = time.time()
        self.last_display_time = time.time()
        
        # Buffer untuk batch processing
        self.message_buffer = deque(maxlen=10000)
        
        # Regex untuk parsing (compiled untuk performa)
        self.message_pattern = re.compile(r'\[(\d+),([0-9A-F]+)\]')
        
    async def register_client(self, websocket):
        """Register new client connection"""
        self.clients.add(websocket)
        client_ip = websocket.remote_address[0]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client connected: {client_ip}")
        
        try:
            async for message in websocket:
                await self.handle_batched_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Client disconnected: {client_ip}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection error: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def handle_batched_message(self, websocket, batch_message):
        """Handle batched messages from ESP32"""
        try:
            self.total_batches += 1
            
            # Parse multiple messages dalam satu batch
            messages = self.parse_batch(batch_message)
            
            if messages:
                self.total_messages += len(messages)
                
                # Process setiap message dalam batch
                for timestamp, hex_data in messages:
                    self.process_single_message(timestamp, hex_data)
                
                # Display statistics
                await self.display_statistics()
                
                # Send ACK untuk setiap 10 batch
                if self.total_batches % 10 == 0:
                    ack_message = f"[ACK_BATCH,{self.total_batches},{len(messages)}]"
                    try:
                        await websocket.send(ack_message)
                    except:
                        pass  # Ignore send errors
                        
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error processing batch: {e}")
    
    def parse_batch(self, batch_message):
        """Parse batched message format: [ts1,hex1][ts2,hex2]..."""
        try:
            # Find all matches menggunakan regex
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
            print(f"Parse error: {e}")
            return []
    
    def process_single_message(self, timestamp, hex_data):
        """Process individual message"""
        try:
            # Convert hex to decimal
            decimal_value = int(hex_data, 16)
            
            # Store di buffer untuk processing lebih lanjut
            self.message_buffer.append({
                'timestamp': timestamp,
                'hex': hex_data,
                'decimal': decimal_value,
                'received_time': time.time()
            })
            
        except ValueError:
            # Invalid hex data
            pass
    
    async def display_statistics(self):
        """Display performance statistics"""
        current_time = time.time()
        
        # Display setiap 1 detik atau setiap 100 messages
        if (current_time - self.last_display_time >= 1.0) or (self.total_messages % 1000 == 0):
            elapsed_time = current_time - self.start_time
            
            if elapsed_time > 0:
                msg_rate = self.total_messages / elapsed_time
                batch_rate = self.total_batches / elapsed_time
                avg_batch_size = self.total_messages / self.total_batches if self.total_batches > 0 else 0
                
                # Get latest message untuk display
                if self.message_buffer:
                    latest = self.message_buffer[-1]
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Total: {self.total_messages:,} | "
                          f"Rate: {msg_rate:.1f} msg/s | "
                          f"Batches: {self.total_batches} | "
                          f"Avg/batch: {avg_batch_size:.1f} | "
                          f"Latest: TS:{latest['timestamp']}, "
                          f"Hex:{latest['hex']}, Dec:{latest['decimal']}")
                
                self.last_display_time = current_time
    
    async def broadcast_message(self, message):
        """Broadcast message to all connected clients"""
        if self.clients:
            disconnected = []
            for client in self.clients:
                try:
                    await client.send(message)
                except:
                    disconnected.append(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.clients.discard(client)
    
    async def start_server(self):
        """Start optimized WebSocket server"""
        print(f"Starting optimized WebSocket server on {self.host}:{self.port}")
        print("Optimized for 860 SPS with batching support")
        print("Waiting for ESP32 connections...")
        
        # Server settings untuk high-frequency data
        server_settings = {
            'compression': None,  # Disable compression untuk speed
            'max_size': 2**20,    # 1MB max message size
            'max_queue': 100,     # Message queue size
            'ping_interval': 20,  # Ping every 20 seconds
            'ping_timeout': 10,   # Ping timeout 10 seconds
        }
        
        async with websockets.serve(
            self.register_client,
            self.host,
            self.port,
            **server_settings
        ):
            await asyncio.Future()  # Run forever

    def get_statistics(self):
        """Get current performance statistics"""
        elapsed = time.time() - self.start_time
        return {
            'total_messages': self.total_messages,
            'total_batches': self.total_batches,
            'elapsed_time': elapsed,
            'message_rate': self.total_messages / elapsed if elapsed > 0 else 0,
            'batch_rate': self.total_batches / elapsed if elapsed > 0 else 0,
            'buffer_size': len(self.message_buffer)
        }

async def main():
    server = OptimizedWebSocketServer()
    await server.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")