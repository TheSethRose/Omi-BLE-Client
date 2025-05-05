#!/usr/bin/env python3
"""
Omi BLE Audio Transcription Pipeline

Main entry point for the BLE audio capture and transcription system.
Connects to a Friend BLE device, captures audio, and performs real-time transcription.
"""

import asyncio
import uuid
import os
import sys
from dotenv import load_dotenv
from modules.transcription import Transcriber, SpeechDetector
from modules.bluetooth import discover_ble_devices, connect_to_device
from modules.database import Database

# Always load .env at startup
load_dotenv()

async def run():
    """Entry-point: connect to BLE device and stream speech to NVIDIA Parakeet."""
    print("Loading NVIDIA Parakeet model…")
    try:
        db = Database()
        print("Connected to PostgreSQL database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    def on_transcription(text, meta):
        # Store each utterance with a unique ID
        rec_id = str(uuid.uuid4())
        db.create("utterances", id=rec_id, document=text, metadata=meta)

    transcriber = Transcriber(on_transcription=on_transcription)
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
