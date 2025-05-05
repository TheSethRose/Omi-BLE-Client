"""Microphone audio capture for macOS/Unix using sounddevice.

Captures 16-kHz, 16-bit mono PCM from the default input device and feeds raw
PCM bytes to a provided SpeechDetector instance via `detector.add_pcm()`.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Hz
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (int16)


class MicrophoneCapture:
    """Continuously capture microphone audio and forward PCM to detector."""

    def __init__(self, detector, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS, device: Optional[int] = None):
        self.detector = detector
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.stream: Optional[sd.InputStream] = None
        self.running = False
        # 10-ms blocks to match VAD frame duration
        self.blocksize = int(self.sample_rate * 0.01)  # 160 samples at 16 kHz

    def _callback(self, indata: np.ndarray, _frames: int, _time, _status):
        """sounddevice callback -> forward PCM bytes"""
        if not self.running:
            return
        # Ensure int16
        pcm_bytes = indata.tobytes()
        if pcm_bytes:
            self.detector.add_pcm(pcm_bytes)

    def start(self):
        if self.running:
            return
        self.running = True
        # Print available devices and selected device info
        print("[DEBUG] Listing available audio input devices:")
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        for idx, dev in enumerate(input_devices):
            print(f"  [{idx}] {dev['name']} (inputs: {dev['max_input_channels']})")
        if self.device is None:
            while True:
                try:
                    selection = input(f"Select input device index [0-{len(input_devices)-1}] (default: 0): ")
                    if selection.strip() == "":
                        device_idx = 0
                    else:
                        device_idx = int(selection)
                    if 0 <= device_idx < len(input_devices):
                        self.device = input_devices[device_idx]['name']
                        dev_info = input_devices[device_idx]
                        print(f"[INFO] Microphone capture using device: {dev_info['name']} (index: {device_idx})")
                        break
                    else:
                        print(f"[WARN] Invalid index. Please enter a number between 0 and {len(input_devices)-1}.")
                except ValueError:
                    print("[WARN] Please enter a valid number.")
        else:
            dev_info = sd.query_devices(self.device)
            print(f"[INFO] Microphone capture using device: {dev_info['name']} (index: {self.device})")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.blocksize,
            callback=self._callback,
            device=self.device,
        )
        self.stream.start()
        print("[INFO] Microphone capture started (Ctrl-C to stop)…")
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[INFO] Microphone capture stopped by user")
        finally:
            self.stop()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("[INFO] Microphone capture stopped")
