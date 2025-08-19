"""
Generate mock STM-format EKG-like signals for testing.

Output format per sample (STM style):
  t_ms_hex,CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8,CH9;

Where values are uppercase hex without prefix, timestamp is milliseconds in hex.

Specs:
- 400 SPS (2.5 ms/sample) for 5 seconds (2000 samples)
- 9 channels (12-bit 0..4095, centered at 2048)
- Channels mapping (example set):
  CH1: sine
  CH2: sine_noise
  CH3: ecg
  CH4: ecg_noise
  CH5: square
  CH6: square_noise
  CH7: sawtooth
  CH8: sawtooth_noise
  CH9: triangle_noise

Usage:
  python generate_mock_stm.py --seed 42 --outfile mock_stm_400sps_5s_seed42.csv
If --outfile omitted, a default name with the seed will be used.
"""

import os
import sys
import math
import argparse
import numpy as np


SPS = 400
DURATION_S = 50.0
NUM_SAMPLES = int(SPS * DURATION_S)
CENTER = 2048.0
MAX_VAL = 4095.0


def clip12(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, MAX_VAL)


def sine_wave(t: np.ndarray, freq: float = 1.0, amplitude: float = 800.0) -> np.ndarray:
    return CENTER + amplitude * np.sin(2.0 * math.pi * freq * t)


def square_wave(t: np.ndarray, freq: float = 1.0, amplitude: float = 800.0, duty: float = 0.5) -> np.ndarray:
    phase = (t * freq) % 1.0
    return CENTER + amplitude * np.where(phase < duty, 1.0, -1.0)


def sawtooth_wave(t: np.ndarray, freq: float = 1.0, amplitude: float = 800.0) -> np.ndarray:
    phase = (t * freq) % 1.0
    # Map phase [0,1) to [-1,1)
    saw = 2.0 * phase - 1.0
    return CENTER + amplitude * saw


def triangle_wave(t: np.ndarray, freq: float = 1.0, amplitude: float = 800.0) -> np.ndarray:
    phase = (t * freq) % 1.0
    tri = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    return CENTER + amplitude * tri


def ecg_like(t: np.ndarray, hr_hz: float = 1.0) -> np.ndarray:
    """Very simple synthetic ECG using Gaussian P, QRS, T waves each beat.
    hr_hz: beats per second (1.0 = 60 BPM)
    """
    # Positions (seconds into beat)
    beat_phase = (t * hr_hz) % 1.0
    # Convert to time within beat [0,1)
    tb = beat_phase

    def gauss(mu, sigma, amp):
        return amp * np.exp(-0.5 * ((tb - mu) / sigma) ** 2)

    # P-wave (~0.2 s), small
    p = gauss(0.2, 0.03, 120.0)
    # QRS complex (~0.0-0.04 s), sharp high amp
    qrs = gauss(0.0, 0.015, 1000.0)
    # T-wave (~0.35 s), medium
    tw = gauss(0.35, 0.06, 220.0)

    return CENTER + (p - 40.0) + qrs + (tw - 60.0)


def add_noise(x: np.ndarray, noise_std: float = 60.0, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return x + rng.normal(0.0, noise_std, size=x.shape)


def build_channels(t: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    # Base signals
    ch1 = sine_wave(t, freq=1.0, amplitude=800.0)
    ch3 = ecg_like(t, hr_hz=1.0)
    ch5 = square_wave(t, freq=1.0, amplitude=700.0, duty=0.5)
    ch7 = sawtooth_wave(t, freq=1.0, amplitude=700.0)
    ch9 = triangle_wave(t, freq=1.0, amplitude=650.0)

    # Noisy counterparts
    ch2 = add_noise(ch1, noise_std=70.0, rng=rng)
    ch4 = add_noise(ch3, noise_std=70.0, rng=rng)
    ch6 = add_noise(ch5, noise_std=70.0, rng=rng)
    ch8 = add_noise(ch7, noise_std=70.0, rng=rng)
    ch9n = add_noise(ch9, noise_std=70.0, rng=rng)  # choose noisy triangle for CH9

    channels = [ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9n]
    # Clip and round to nearest int within 0..4095
    channels = [np.rint(clip12(c)).astype(np.int64) for c in channels]
    return channels


def ms_timestamp_hex(n: int, sps: int) -> str:
    # Use rounded cumulative ms so deltas are 2 or 3 ms (avg 2.5 ms)
    t_ms = int(round(n * 1000.0 / float(sps)))
    return f"{t_ms:06X}"


def write_stm_csv(out_path: str, channels: list[np.ndarray]) -> None:
    with open(out_path, "w", encoding="ascii", newline="") as f:
        # Write data rows
        for i in range(NUM_SAMPLES):
            ts_hex = ms_timestamp_hex(i, SPS)
            vals_hex = ",".join(f"{int(ch[i]):03X}" for ch in channels)
            f.write(ts_hex)
            f.write(",")
            f.write(vals_hex)
            f.write(";\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate STM-format mock EKG signals (400 SPS, 5s, 9 channels)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise (default: 42)")
    parser.add_argument("--outfile", type=str, default="", help="Output CSV path (default under current dir)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Time vector
    t = np.arange(NUM_SAMPLES, dtype=np.float64) / float(SPS)

    # Build channels
    channels = build_channels(t, rng)

    # Determine output path
    out_name = args.outfile.strip()
    if not out_name:
        out_name = f"mock_stm_400sps_5s_seed{args.seed}.csv"

    out_dir = os.path.dirname(out_name)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    write_stm_csv(out_name, channels)
    print(f"Generated: {out_name}  (samples={NUM_SAMPLES}, sps={SPS})")


if __name__ == "__main__":
    main()


