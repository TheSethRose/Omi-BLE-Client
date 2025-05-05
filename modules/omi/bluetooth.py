#!/usr/bin/env python3
"""
Bluetooth module for the Omi BLE Audio project.
Handles BLE device discovery, connection, and audio capture.
"""

import asyncio
import opuslib
from datetime import datetime
from bleak import BleakClient, BleakScanner

# Nordic UART Service UUIDs
NORDIC_UART_SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
NORDIC_UART_TX_CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"  # Write to this characteristic
NORDIC_UART_RX_CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"  # Notifications come from this characteristic

# Audio settings - must match transcription.py
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1


class FrameProcessor:
    """Assembles multi-packet Opus frames exactly like the original devkit code."""
    def __init__(self, sample_rate: int, channels: int):
        self.decoder = opuslib.Decoder(sample_rate, channels)
        self.last_packet_index = -1   # 16-bit rolling index from device
        self.last_frame_id = -1       # byte 2 in packet: position within current frame (0 = first)
        self.pending = bytearray()    # bytes collected for current frame
        self.frames = []              # completed encoded Opus frames
        self.lost = 0

    def store_packet(self, data: bytes):
        index = data[0] + (data[1] << 8)  # two-byte packet counter
        frame_id = data[2]
        content = data[3:]

        # Start of stream detection
        if self.last_packet_index == -1 and frame_id == 0:
            self.last_packet_index = index
            self.last_frame_id = frame_id
            self.pending = bytearray(content)
            return

        # Still waiting for the first frame start
        if self.last_packet_index == -1:
            return

        # Loss detection: non-contiguous packet or frame id jump
        if index != (self.last_packet_index + 1) % 65536 or (
            frame_id != 0 and frame_id != self.last_frame_id + 1
        ):
            print("Lost packets, dropping current frame")
            self.last_packet_index = -1
            self.pending.clear()
            self.lost += 1
            return

        # New frame starts when frame_id == 0
        if frame_id == 0:
            # store completed frame
            if self.pending:
                self.frames.append(bytes(self.pending))
            self.pending = bytearray(content)
        else:
            self.pending.extend(content)

        self.last_frame_id = frame_id
        self.last_packet_index = index

    def pop_pcm(self) -> bytearray:
        """Decode any completed frames to PCM and return as bytearray."""
        if self.pending and self.last_frame_id != 0:
            # waiting for next frame start; nothing to decode yet
            pass
        pcm = bytearray()
        frame_size = 960  # 20 ms @ 48 kHz; works for 16 kHz decoder too
        while self.frames:
            frame = self.frames.pop(0)
            try:
                pcm.extend(self.decoder.decode(frame, frame_size))
            except Exception as e:
                print(f"Error decoding frame: {e}")
        return pcm


class AudioCapture:
    """Receives BLE packets, decodes them, forwards PCM to SpeechDetector."""

    def __init__(self, detector):
        self.frame_processor = FrameProcessor(SAMPLE_RATE, CHANNELS)
        self.detector = detector

    def add_packet(self, data: bytes) -> None:
        """Store packet, decode any ready frames, feed PCM to detector."""
        self.frame_processor.store_packet(data)
        pcm = self.frame_processor.pop_pcm()
        if pcm:
            self.detector.add_pcm(pcm)


async def discover_ble_devices():
    """Discover and list BLE devices, return the Friend device if found."""
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    
    if not devices:
        print("No BLE devices found. Exiting.")
        return None
    
    print("Discovered devices:")
    for dev in devices:
        print(f"  {dev.name or 'Unknown'} ({dev.address})")
    
    # Find device with "Friend" in the name
    omi_device = next((d for d in devices if d.name and "Friend" in d.name), None)
    
    if omi_device is None:
        print("Friend device not found. Make sure it is powered on and advertising.")
        return None
    
    return omi_device


async def connect_to_device(device, speech_detector):
    """Connect to BLE device and set up audio streaming."""
    print(f"Connecting to {device.name} ({device.address})...")
    
    # Create audio capture with speech detector
    audio_capture = AudioCapture(speech_detector)
    
    def notification_handler(_: int, data: bytes) -> None:
        """Callback for BLE notifications: forward raw packet to decoder."""
        audio_capture.add_packet(data)
    
    async with BleakClient(device.address) as client:
        print(f"Connected: {client.is_connected}")
        
        # Enable audio notifications and command device to start streaming
        await client.start_notify(NORDIC_UART_RX_CHAR_UUID, notification_handler)
        await client.write_gatt_char(NORDIC_UART_TX_CHAR_UUID, b"mic capture 0\n")
        
        print("Started streaming audio. Speak into the Friend mic! (Ctrl-C to stop)")
        
        try:
            # Keep connection alive until interrupted
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping audio stream...")
