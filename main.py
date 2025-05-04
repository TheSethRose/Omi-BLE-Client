#!/usr/bin/env python3
"""
Omi BLE Audio Transcription Pipeline

Main entry point for the BLE audio capture and transcription system.
Connects to a Friend BLE device, captures audio, and performs real-time transcription.
"""

import asyncio
from modules.transcription import Transcriber, SpeechDetector
from modules.bluetooth import discover_ble_devices, connect_to_device


async def run():
    """Entry-point: connect to BLE device and stream speech to NVIDIA Parakeet."""
    print("Loading NVIDIA Parakeet model…")
    transcriber = Transcriber()
    detector = SpeechDetector(transcriber)

    try:
        # Discover and connect to BLE device
        device = await discover_ble_devices()
        if not device:
            return

        # Connect to device and start audio streaming
        await connect_to_device(device, detector)
    finally:
        transcriber.stop()


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nScript terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
