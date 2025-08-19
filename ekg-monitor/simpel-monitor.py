"""
Simple console monitor for STM32 CSV+semicolon data stream.

Reads serial frames terminated by ';' in the format:
  t_ms_hex,CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8,CH9;

Prints one line per sample to stdout. By default prints timestamp_seconds and
all 9 channels in decimal CSV. Optionally select a single channel (1-9).

Debug/info lines from firmware starting with '#' (newline-terminated) are
printed to stderr as-is.
"""

import sys
import time
import argparse
from typing import Optional

import serial
import serial.tools.list_ports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple STM EKG console monitor")
    parser.add_argument("--port", help="Serial port (e.g., COM5 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=250000, help="Baudrate (default: 250000)")
    parser.add_argument("--channel", type=int, default=0, help="Channel to print (1-9), 0=all (default)")
    parser.add_argument("--no-ts", action="store_true", help="Do not print timestamp seconds column")
    return parser.parse_args()


def auto_select_port() -> Optional[str]:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    return ports[0].device


def main() -> None:
    args = parse_args()

    port = args.port or auto_select_port()
    if not port:
        print("No serial ports found. Use --port to specify.", file=sys.stderr)
        sys.exit(1)

    if args.channel < 0 or args.channel > 9:
        print("--channel must be 0 (all) or 1-9", file=sys.stderr)
        sys.exit(1)

    try:
        ser = serial.Serial(port, args.baud, timeout=0.1)
    except Exception as e:
        print(f"Failed to open {port}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to {port} @ {args.baud} baud", file=sys.stderr)
    if args.channel == 0:
        print("Output: timestamp_seconds, CH1..CH9 (decimal)", file=sys.stderr)
    else:
        print(f"Output: timestamp_seconds, CH{args.channel} (decimal)", file=sys.stderr)

    buffer = bytearray()
    try:
        while True:
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                buffer.extend(chunk)

            # Handle newline-terminated debug lines
            while True:
                nl_pos = buffer.find(b"\n")
                if nl_pos == -1:
                    break
                line = buffer[:nl_pos]
                buffer = buffer[nl_pos + 1 :]
                if line.endswith(b"\r"):
                    line = line[:-1]
                if not line:
                    continue
                try:
                    text = line.decode(errors="ignore").strip()
                except Exception:
                    continue
                if text.startswith('#'):
                    print(text, file=sys.stderr)
                # else: keep non-debug newlines ignored here

            # Handle semicolon-terminated data frames
            while True:
                semi_pos = buffer.find(b";")
                if semi_pos == -1:
                    break
                frame = buffer[:semi_pos]
                buffer = buffer[semi_pos + 1 :]
                if not frame:
                    continue

                try:
                    parts = frame.decode("ascii", errors="ignore").split(",")
                    if len(parts) != 10:
                        # malformed, skip
                        continue
                    t_ms = int(parts[0].strip(), 16)
                    t_sec = t_ms / 1000.0
                    ch_vals = [int(p.strip(), 16) for p in parts[1:]]
                except Exception:
                    continue

                # Print one line per sample
                if args.channel == 0:
                    if args.no_ts:
                        print(",".join(str(v) for v in ch_vals), flush=True)
                    else:
                        print(f"{t_sec:.6f}," + ",".join(str(v) for v in ch_vals), flush=True)
                else:
                    idx = args.channel - 1
                    if 0 <= idx < len(ch_vals):
                        if args.no_ts:
                            print(str(ch_vals[idx]), flush=True)
                        else:
                            print(f"{t_sec:.6f},{ch_vals[idx]}", flush=True)

            # Avoid busy loop
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopped by user", file=sys.stderr)
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


