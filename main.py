#!/usr/bin/env python3
"""
Transcription Orchestrator

Selects capture source (BLE Omi DevKit or local microphone) based on
CAPTURE_SOURCE env var. Streams audio to SpeechDetector -> Transcriber and
persists utterances grouped into conversations.
"""

import asyncio
import uuid
import os
from datetime import datetime

from dotenv import load_dotenv
from modules.transcription import Transcriber, SpeechDetector
from modules.database import Database
from modules.conversation import ConversationManager

# ---------------------------------------------------------------------------
# Environment & capture source selection
# ---------------------------------------------------------------------------
load_dotenv()

CAPTURE_SOURCE = os.getenv("CAPTURE_SOURCE", "omi").lower()

if CAPTURE_SOURCE == "omi":
    from modules.omi.bluetooth import discover_ble_devices, connect_to_device  # type: ignore
elif CAPTURE_SOURCE == "microphone":
    from modules.microphone.capture import MicrophoneCapture  # type: ignore
else:
    raise ValueError("CAPTURE_SOURCE must be 'omi' or 'microphone'")


# ---------------------------------------------------------------------------
# Helper runners for each capture backend
# ---------------------------------------------------------------------------
async def _run_ble(detector: SpeechDetector):
    """Discover Friend device and stream audio via BLE."""
    device = await discover_ble_devices()
    if device:
        await connect_to_device(device, detector)
    else:
        print("No suitable BLE device found.")


def _run_microphone(detector: SpeechDetector):
    """Stream audio from local microphone."""
    mic = MicrophoneCapture(detector)
    mic.start()


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading NVIDIA Parakeet model…")
    try:
        db = Database()
        print("Connected to PostgreSQL database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    # Conversation grouping
    conversation_manager = ConversationManager(db)

    def on_transcription(text: str, meta: dict):
        # Determine conversation ID
        conv_id = conversation_manager.get_conversation_for_utterance(meta["timestamp"])
        conversation_manager.update_conversation_metadata(conv_id, datetime.fromisoformat(meta["timestamp"].replace(' ', 'T')))
        meta["conversation_id"] = conv_id

        rec_id = str(uuid.uuid4())
        db.create("utterances", id=rec_id, document=text, metadata=meta)

    transcriber = Transcriber(on_transcription=on_transcription)
    detector = SpeechDetector(transcriber)

    try:
        if CAPTURE_SOURCE == "omi":
            asyncio.run(_run_ble(detector))
        else:
            _run_microphone(detector)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        transcriber.stop()


if __name__ == "__main__":
    main()
