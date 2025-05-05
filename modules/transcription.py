#!/usr/bin/env python3
"""
Transcription module for the Omi BLE Audio project.
Handles speech detection, VAD, and transcription using NVIDIA Parakeet TDT model.
"""

import os
import math
import time
import threading
import warnings
import numpy as np
import soundfile as sf
import tempfile
import nemo.collections.asr as nemo_asr
import webrtcvad
from datetime import datetime
from queue import Queue
from pathlib import Path
import logging
from tqdm import tqdm
from collections import deque
import re
import uuid

# Reduce Nemo verbosity and tqdm progress bars
os.environ["NEMO_LOG_LEVEL"] = "ERROR"
for _name in ("nemo", "nemo_logging"):
    logging.getLogger(_name).setLevel(logging.CRITICAL + 1)
# Disable all logs globally except CRITICAL
logging.disable(logging.CRITICAL)
tqdm.write = lambda *args, **kwargs: None  # Disable tqdm output
tqdm.tqdm = lambda *args, **kwargs: args[0]  # Disable progress bar completely

# Audio settings
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (16-bit)

# VAD / streaming parameters
FRAME_DURATION_MS = 10  # webrtcvad frame size must be 10/20/30 ms
BYTES_PER_SAMPLE = SAMPLE_WIDTH  # 2 bytes for 16-bit audio
FRAME_BYTES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * BYTES_PER_SAMPLE  # 320 bytes
SILENCE_THRESHOLD_FRAMES = 20  # flush if this many frames of silence observed (~600 ms)
MAX_BUFFER_SECONDS = 30.0  # flush if buffer exceeds this duration (match NVIDIA Parakeet's window)
OVERLAP_SECONDS = 0.0  # no overlap to prevent duplicate segments

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)


class Transcriber:
    """
    Handles speech-to-text transcription with NVIDIA Parakeet TDT model.
    Uses a dedicated background thread to process audio segments.
    """

    SUPPRESSION_PHRASES = [
        "thank you", "thanks for", "subscribe", "for watching",
        "see you", "next video", "視聴", "ご視聴", "opening theme",
    ]
    GARBAGE_LINES = [
        "a", "aa", "ah", "ha", "h", "uh", "oh", "mmm", "mm", "hm", "okay"
    ]
    # Optimal logprob threshold for NVIDIA Parakeet confidence; filter out repetitions/garbage
    AVG_LOGPROB_THRESHOLD = -1.1  # Balanced: allows valid speech, filters most junk
    COMPRESSION_RATIO_THRESHOLD = 2.4
    NO_SPEECH_THRESHOLD = 0.6

    def __init__(self, on_transcription=None):
        # Suppress harmless FP16 warning on CPU
        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

        # Detect Apple Silicon (arm64) or not
        import platform
        is_apple_silicon = (platform.system() == "Darwin" and platform.machine() == "arm64")

        if is_apple_silicon:
            # Apple Silicon: Use CPU for maximum compatibility (MPS backend is not fully supported by NVIDIA Parakeet)
            device = "cpu"
            print("[INFO] Apple Silicon detected. Running ASR model on CPU for compatibility.")
        else:
            # Non-Apple Silicon: Use CUDA if available, else CPU
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            print(f"[INFO] Non-Apple Silicon detected. Using device: {device}")

        # Load NVIDIA Parakeet TDT model (English, fast, accurate)
        self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        # Ensure model runs on desired device (CPU or MPS). Parakeet runs fp32 on CPU fine.
        if device != "cpu":
            try:
                self.model = self.model.to(device)
            except Exception:
                print("[WARN] Could not move model to device, falling back to CPU")

        self.queue = Queue()
        self.running = True
        self.last_printed = ""
        self.previous_text = ""
        self.recent_texts = deque(maxlen=10)  # track recent utterances to avoid duplicates
        # Callback for handling valid utterances
        self.on_transcription = on_transcription
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.start()

    def _should_skip(self, text: str) -> bool:
        t = text.strip().lower()
        # Skip empty, garbage, or filler lines
        if not t or t in self.GARBAGE_LINES:
            return True
        # Skip if too short (less than 3 chars or fewer than 1 word)
        if len(t) < 3 or len(t.split()) < 1:
            return True
        # Skip if any word is repeated more than 4 times
        if self._is_repetitive(t, max_repeat=4):
            return True
        # Skip if any suppression phrase is present
        for phrase in self.SUPPRESSION_PHRASES:
            if phrase in t:
                return True
        return False

    @staticmethod
    def _is_repetitive(text, max_repeat=3):
        """Return True if any word is repeated more than max_repeat times."""
        words = text.lower().split()
        return any(words.count(word) > max_repeat for word in set(words))

    def _needs_retry(self, result):
        """Check if we need to retry with a higher temperature."""
        if not result["segments"]:
            return True

        for segment in result["segments"]:
            if segment["compression_ratio"] > self.COMPRESSION_RATIO_THRESHOLD:
                return True  # Too repetitive

            if segment["avg_logprob"] < self.AVG_LOGPROB_THRESHOLD:
                return True  # Too uncertain

            if segment["no_speech_prob"] > self.NO_SPEECH_THRESHOLD and \
               segment["avg_logprob"] < self.AVG_LOGPROB_THRESHOLD:
                return False  # It's silence, no need to retry

        return False

    def _process_queue(self):
        while self.running:
            try:
                if self.queue.empty():
                    time.sleep(0.1)
                    continue

                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    continue

                # Unpack the queue item
                if isinstance(item, tuple):
                    if len(item) == 3:
                        audio_np, start_time, context = item
                    elif len(item) == 2:
                        audio_np, start_time = item
                        context = None
                    else:
                        audio_np = item[0]
                        start_time = time.time()
                        context = None
                else:
                    audio_np, start_time, context = item, time.time(), None

                # Write audio to a temporary WAV file (16kHz mono)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                    sf.write(tmp_wav.name, audio_np, SAMPLE_RATE)
                    try:
                        trans_results = self.model.transcribe([tmp_wav.name], timestamps=True, verbose=False)
                    except Exception as e:
                        logging.error(f"Transcription error: {e}")
                        self.queue.task_done()
                        continue

                if not trans_results:
                    self.queue.task_done()
                    continue

                # Parakeet returns list of Hypothesis or strings
                hyp = trans_results[0]
                joined = hyp.text if hasattr(hyp, 'text') else str(hyp)

                if joined and not self._should_skip(joined):
                    # For Parakeet, use hypothesis.score as confidence
                    conf_score = getattr(hyp, 'score', 0.0)
                    # Simple confidence filter: skip if score (neg logprob) < -1.0
                    if conf_score < -1.0:
                        self.queue.task_done()
                        continue

                    # Filter out single-word utterances (<=1 non-filler word)
                    if len(re.findall(r"[a-zA-Z]+", joined)) < 2:
                        self.queue.task_done()
                        continue

                    # Deduplicate: normalize and check against recent_texts
                    norm = re.sub(r"[^a-zA-Z]+", " ", joined.lower()).strip()
                    if norm in self.recent_texts:
                        self.queue.task_done()
                        continue
                    self.recent_texts.append(norm)

                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    latency = time.time() - start_time
                    line = f"[{timestamp}] score={conf_score:.2f} latency={latency:.2f}s | {joined}"
                    print(line)
                    # Invoke callback for storing utterance
                    if self.on_transcription:
                        self.on_transcription(joined, {"timestamp": timestamp, "score": conf_score, "latency": latency})
                    self.last_printed = joined
                    self.previous_text = joined
                    # Clear queue to avoid backlog duplicates
                    with self.queue.mutex:
                        self.queue.queue.clear()

                self.queue.task_done()

            except Exception as e:
                print(f"Error in transcription thread: {e}")
                self.queue.task_done()

    def add_audio(self, audio, context=None) -> None:
        """Queue a float32 numpy array with optional context for transcription."""
        if isinstance(audio, tuple) and len(audio) == 2:
            audio_np, timestamp = audio
            self.queue.put((audio_np, timestamp, context))
        else:
            self.queue.put((audio, time.time(), context))

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()


class SpeechDetector:
    """Detect speech in audio using WebRTC VAD."""

    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
        self.buffer = bytearray()
        self.tail = bytearray()
        self.silence_count = 0
        self.context = ""  # Store previous transcription as context
        self.silence_frames = SILENCE_THRESHOLD_FRAMES
        self.max_bytes = int(MAX_BUFFER_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)
        self.overlap_bytes = int(OVERLAP_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)

    def _flush(self) -> None:
        """Flush current speech buffer to the transcriber with overlap."""
        if not self.buffer:
            return

        pcm = bytes(self.buffer)
        # If we have an overlap from previous chunk, prepend it
        if self.tail:
            pcm = self.tail + pcm

        # Keep last overlap for next chunk
        self.tail = self.buffer[-self.overlap_bytes:] if len(self.buffer) >= self.overlap_bytes else self.buffer
        self.buffer.clear()
        self.silence_count = 0

        # Convert to numpy and send to transcriber
        audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        logging.debug(f"Speech segment captured ({len(audio_np)/SAMPLE_RATE:.2f}s), sending to transcriber.")

        # Pass audio with context to transcriber
        self.transcriber.add_audio((audio_np, time.time()), self.context)

        # Update context from transcriber's last output
        if self.transcriber.last_printed:
            self.context = self.transcriber.last_printed

    def add_pcm(self, pcm_data: bytes) -> None:
        """Process raw PCM bytes from decoder."""
        if not pcm_data:
            return

        i = 0
        while i + FRAME_BYTES <= len(pcm_data):
            frame = pcm_data[i : i + FRAME_BYTES]
            i += FRAME_BYTES

            try:
                is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            except Exception:
                is_speech = False

            if is_speech:
                self.buffer.extend(frame)
                self.silence_count = 0
            else:
                self.silence_count += 1
                if self.silence_count >= self.silence_frames or len(self.buffer) >= self.max_bytes:
                    self._flush()

        # Append leftover (< one frame) directly – will be handled next call
        if i < len(pcm_data):
            self.buffer.extend(pcm_data[i:])
