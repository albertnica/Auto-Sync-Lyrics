# AI Lyrics Retriever & Synchronizer

A practical Python tool to generate synchronized LRC lyrics for audio files (FLAC). The project combines multi-model ASR (Whisper) and interpolation logic to produce conservative, high-quality line timestamps and LRC files from plain Genius lyrics.

## Features
- Compare transcriptions from multiple Whisper models and pick the best candidate.
- Identify reliable anchor lines and interpolate timestamps for other lines.
- Preserve and reuse canonical LRCs in a local `.lyrics_db` folder.
- Treat local `.lrc` files placed inside `songs/` as authoritative (moved into `lyrics/` and copied to `.lyrics_db`).
- Optional Genius integration to fetch raw lyrics (requires token in `credentials.json`).
- Per-run logs and per-model diagnostics saved under `.logs/`.

## Quick start

Prerequisites:
- Python 3.10+ (3.13 recommended).
- Git (optional).

1) Clone the repo (or download the files):

```powershell
git clone <repo-url>
cd Automatic-Lyrics-Synchronizer
```

2) Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

3) Install Python dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4) (Optional) Install a CUDA-enabled PyTorch wheel if you plan to use GPU acceleration. Find the right command for your system at https://pytorch.org/ and run it in the activated venv.

```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

5) Prepare folders and assets:
- Put your FLAC files (and optional existing .lrc files) under `songs/`.
- Optionally run [LRCGET](https://github.com/tranxuanthang/lrcget) on the `songs/` folder to search for existing synced lyric files from public sources.
- Create a `credentials.json` at the project root if you want Genius support:

```json
{
  "genius_access_token": "YOUR_GENIUS_TOKEN"
}
```

6) Run the main script:

```powershell
python main.py
```

After a successful run:
- Generated LRCs are written to `lyrics/`.
- Canonical copies are stored in `.lyrics_db/`.
- Logs and raw data for each run appear under `.logs/<timestamp>/`.

## Configuration

Edit `config.py` to tune behavior. Key options:
- `DEBUG`: verbose debug output.
- `SONGS_FOLDER`, `LYRICS_FOLDER`, `LYRICS_DB_FOLDER`, `LOGS_FOLDER`.
- `TRANSCRIBE_MODELS`: list of Whisper model names to try (order matters).

## How it works

1. Create a per-run folder in `.logs/` and initialize logging.
2. Recreate the `lyrics/` output folder and ingest any `.lrc` files found in `songs/` (these are treated as authoritative).
3. For each `.flac` in `songs/`:
   - Try to restore an existing canonical LRC from `.lyrics_db/` using FLAC metadata (title/artist).
   - If not restored, use a local `.txt` (if present) as the lyrics source.
   - If no local lyrics and a Genius token exists, attempt to fetch lyrics (raw text saved to logs only).
   - For each configured ASR model:
     - Transcribe the audio and collect token timestamps where available.
     - Compute candidate anchors for lyric lines and filter by thresholds.
     - Interpolate timestamps for non-anchor lines while avoiding interpolation across long silences.
   - Score each model and choose the best one.
   - Write the final `.lrc` into `lyrics/` and copy into `.lyrics_db/` under a canonical name derived from metadata.

Detailed logs for each step are saved under the run folder to help debugging and tuning.

## Troubleshooting & tips

- If a model fails to run, check `.logs/<run>/` for per-model logs and stack traces.
- If Genius seems to return wrong matches, ensure your FLAC metadata (artist/title) is correct; the system expects a strict title match in candidate scanning by default.
- For large batches, prefer running with a GPU-enabled torch build and increase available resources.

## In future versions

Support custom plain lyrics as an input on /songs.
Personal lyrics database with Supabase

## License

See LICENSE in the repository root.