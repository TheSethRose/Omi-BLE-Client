# Omi BLE Audio Transcription Pipeline

Real-time, hands-free audio transcription using the Omi Dev Kit and OpenAI's Whisper model. Stream room audio wirelessly via Bluetooth Low Energy (BLE) for instant speech-to-text conversion.

## Features

- **Real-time BLE Audio Streaming**: Capture audio from Omi Dev Kit over Bluetooth Low Energy
- **High-Quality Transcription**: Uses OpenAI's Whisper large-v3-turbo model
- **Voice Activity Detection**: Smart speech segmentation using WebRTC VAD
- **Confidence Filtering**: Intelligent filtering based on Whisper's confidence scores
- **Low Latency**: Optimized for real-time transcription with minimal delay
- **Privacy-First**: All processing happens locally on your device

## Requirements

- Python 3.12+
- [Omi Dev Kit](https://www.omi.me/products/omi-dev-kit-2)
- MacOS/Linux/Windows with BLE support

## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Power on your Omi Dev Kit
2. Run the transcription pipeline:
```bash
python main.py
```

3. Speak into the Omi Dev Kit's microphone
4. Transcriptions will appear in real-time and be saved to `transcription.txt`

## Architecture

The pipeline is organized into modular components:

- **main.py**: Entry point and application orchestration
- **modules/bluetooth.py**: BLE device discovery and audio capture
- **modules/transcription.py**: Speech detection and Whisper transcription

## Performance Notes

- Apple Silicon Macs: Uses CPU for maximum Whisper compatibility
- NVIDIA GPUs: Automatically uses CUDA for faster processing
- Other platforms: Falls back to CPU with optimized settings

## License

MIT

## Credits

- Omi Dev Kit by [BasedHardware](https://github.com/BasedHardware/omi)
- OpenAI Whisper for transcription
- WebRTC VAD for voice detection
