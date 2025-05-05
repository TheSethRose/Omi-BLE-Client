# Omi BLE Audio Transcription Pipeline

Real-time, hands-free audio transcription using the Omi Dev Kit and NVIDIA Parakeet TDT. Stream room audio wirelessly via Bluetooth Low Energy (BLE) for instant speech-to-text conversion.

## Features

- **Real-time BLE Audio Streaming**: Capture audio from Omi Dev Kit over Bluetooth Low Energy
- **High-Quality Transcription**: Uses NVIDIA Parakeet TDT 0.6B v2 model
- **Voice Activity Detection**: Smart speech segmentation using WebRTC VAD
- **Confidence Filtering**: Intelligent filtering based on transcription confidence scores
- **Duplicate Prevention**: Removes duplicate transcriptions
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

### PostgreSQL + pgvector via Docker

This project ships with a `docker-compose.yml` that launches a ready-to-use Postgres 16 instance with the pgvector extension pre-installed.

**Note:** This project now uses port `5532` for Postgres (not the default 5432). Update your client settings accordingly.

```bash
# Start the database in the background
docker compose up -d
```

Credentials (also stored in `.env`):

| Variable | Value |
|----------|-------|
| POSTGRES_USER | `omi_user` |
| POSTGRES_PASSWORD | `omi_pass` |
| POSTGRES_DB | `omi_db` |
| Port | `5532` |

The service exposes port **5532** on localhost (container port 5432).

### Environment variables

Create a `.env` (already generated) or export manually:

```bash
export DATABASE_URL="postgresql://omi_user:omi_pass@localhost:5532/omi_db"
```

## Usage

1. Ensure the Postgres container is running (see above).
2. Power on your Omi Dev Kit
3. Run the transcription pipeline:
```bash
python main.py
```

4. Speak into the device and transcriptions will be stored in PostgreSQL with pgvector.

## Architecture

The pipeline is organized into modular components:

- **main.py**: Entry point and application orchestration
- **modules/bluetooth.py**: BLE device discovery and audio capture
- **modules/transcription.py**: Speech detection and NVIDIA Parakeet TDT transcription

## Performance Notes

- Apple Silicon Macs: Uses CPU for maximum compatibility
- NVIDIA GPUs: Automatically uses CUDA for faster processing
- Other platforms: Falls back to CPU with optimized settings

## License

MIT

## Credits

- Omi Dev Kit by [BasedHardware](https://github.com/BasedHardware/omi)
- NVIDIA Parakeet TDT for transcription
- WebRTC VAD for voice detection
