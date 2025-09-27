# AI Lyrics Retriever and Synchronizer

## Project overview

Anchors-Strict LRC Generator is a Python script designed to produce **strict** (timestamped) LRC lyric files for audio (FLAC) by:

* comparing transcriptions from multiple Open AI Whisper models,
* selecting the transcription that provides the best set of reliable **anchors** (lyric lines that can be confidently timestamped),
* interpolating timestamps for the remaining lines while avoiding interpolation across long silences,
* saving generated LRC files to an output folder and to a normalized local DB (`.lyrics_db`),
* optionally fetching lyrics from Genius when local lyrics are not available.

The use of [LRCGET](https://github.com/tranxuanthang/lrcget) is highly recommended; apply it to the `songs` folder before proceeding. This code also depends on the metadata of the music files, which is why [this project](https://github.com/albertnica/Music-Metadata-Handler) may prove useful as well.

---

## Key features

* Compares multiple Whisper models (`TRANSCRIBE_MODELS`) and selects the best by anchor count and score.
* Uses a local lyrics DB `.lyrics_db` to restore or reuse processed LRCs.
* If an LRC file is present inside `songs`, it is treated as **authoritative**: moved into `lyrics` and copied into `.lyrics_db`, overwriting prior DB entries.
* Fetches lyrics from Genius (if configured) and writes raw lyrics only to `.logs` (not to `songs`).
* Avoids interpolating across long silences and applies robust clustering to ignore repeated chorus lines as anchors.
* Produces human-readable logs for each model and a `.logs/.anchors.txt` file listing useful info.

---

## Repository layout

```
project-root
├─ songs                        # put your audio (.flac) + optional local .lrc files here
├─ lyrics                       # .lrc output files
├─ .lyrics_db                   # canonical stored LRCs (Title - Artist.lrc), used for restore
├─ .logs                        # transcripts, model logs, raw lyrics, .anchors.txt
├─ credentials.json             # API tokens (Genius)
├─ anchors_lrc_generator.ipynb  # main script
├─ requirements.txt             # python dependencies
└─ README.md
```

---

## Credentials and Genius API

The script reads `credentials.json` file from the repository root. Example file:

```json
{
  "genius_access_token": "YOUR_GENIUS_API_TOKEN"
}
```

Put your Genius API token in `genius_access_token`.

### How to obtain a Genius API token

1. Create a Genius account (if you don't have one).
2. Go to their API client/Developer section and register a new API client (this process may require approval or notes).
3. Copy the **access token** (sometimes called “API token” or “client access token”) and paste it into `credentials.json`.
4. The script uses `lyricsgenius` which expects a token with privileges to fetch song details and lyrics via the Genius API.

---

## Input conventions & authoritative LRC behavior

* Put `.flac` files inside `songs`. If a `.lrc` with the same base filename exists in `songs`, the script now treats it as **authoritative**:

  * The `.lrc` will be **moved** into `lyrics` (replacing any existing file of the same filename there).
  * The moved `.lrc` will be **copied** into `.lyrics_db` under the canonical name `"<Title> - <Artist>.lrc"` (derived from FLAC metadata or the LRC header tags `[ti:]` and `[ar:]`), overwriting any existing DB entry.
  * If the move fails (cross-device or permissions), the script falls back to copy+remove.
* If there is no `.lrc` and there is a `.txt` with lyrics (same basename), that `.txt` is used as source lyrics (not automatically ingested as authoritative LRC).
* If lyrics are not present locally, and you provided a Genius token, the script will attempt to fetch lyrics from Genius. Raw fetched lyrics are saved only under `.logs` (`<basename>.raw_lyrics.txt`).

---

## Configuration options (script-level constants)

The script contains a `CONFIG` block of constants near the top. Important options you may tune:

* `DEBUG` (bool): enable verbose debug prints.
* Folder names: `SONGS_FOLDER`, `LYRICS_FOLDER`, `LYRICS_DB_FOLDER`, `LOGS_FOLDER`.
* `TRANSCRIBE_MODELS` (list): the Whisper model names to compare (e.g., `["large-v3", "large-v3-turbo"]`).
* Interpolation and silence thresholds:

  * `MIN_SILENCE_DURATION` — minimum gap to consider a silence.
  * `LONG_SILENCE_THRESHOLD` — silence length that blocks interpolation across it.
  * `THRESH_ANCHOR` — minimal similarity score to accept a candidate anchor.
  * `MIN_OVERLAP` — minimal fraction of lyric words present in the matched transcription window.
  * `MIN_ANCHOR_SPACING` — minimum seconds between anchors to avoid clustered anchors.
* Timestamp fallback/progression:

  * `MIN_LINE_PROGRESSION`, `FALLBACK_SPACING`, `FIRST_LINE_WEIGHT`.

Adjust these values to tune precision vs recall of anchors and interpolation behavior.

---

## How the script works (high-level flow)

1. **Initialization**: folders are created/cleaned (`lyrics` recreated, `.lyrics_db` preserved).
2. **Ingest LRCs in `songs`**: any `.lrc` present in `songs` are moved to `lyrics` and copied into `.lyrics_db`.
3. **Clean songs**: non-audio files in `songs` are removed (only `.flac`, `.mp3`, `.wav` preserved).
4. **Loop over `.flac` files**:

   * Try early restore from `.lyrics_db` using FLAC metadata (title/artist).
   * If `.txt` lyrics exist, use them as source lyrics.
   * If not and Genius is configured, try to fetch lyrics from Genius.
   * Split lyrics into lines, cluster repeated lines (chorus repeats) and exclude repeated clusters from anchor candidates.
   * For each Whisper model in `TRANSCRIBE_MODELS`:

     * Transcribe using Whisper (attempt `word_timestamps` if supported).
     * Clean token timestamps, detect long silences.
     * Compute anchor candidates for each lyric line. Accept anchors by thresholds (score >= `THRESH_ANCHOR`, overlap >= `MIN_OVERLAP`, spacing).
     * Save per-model logs (`.log`, `.transcript.txt`, `.whisper.ts.txt`).
     * Compute interpolated final times for all lines using anchors and avoiding long silences.
   * Choose the best model (most anchors, then highest score_sum).
   * Write the final `.lrc` into `lyrics` and copy into `.lyrics_db` if FLAC metadata exists (or canonicalized name).
5. **Write `.logs/.anchors.txt`** listing files containing the song basename and the result (chosen model name or a failure tag like “all models failed” / “lyrics retrieving failed”).

---

## Advanced / Tuning suggestions

* Change `TRANSCRIBE_MODELS` to smaller models to speed up processing (`base`, `small`) or larger ones for higher accuracy.
* Tune `THRESH_ANCHOR` and `MIN_OVERLAP` to be more/less permissive.
* Add an `INGEST_OVERWRITE` boolean in configuration to toggle whether `.lrc` in `songs` should overwrite DB or just copy.
* Add versioned backups of `.lyrics_db` on each overwrite for safety.

---

## Installation

1. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt

2. Install PyTorch with CUDA support (adjust CUDA version as needed):
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129