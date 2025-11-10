"""
modules/ingest.py
Ingest LRC files found next to songs/ into `lyrics/` and the canonical lyrics DB.

Responsibilities:
- Move LRCs from `songs/` to `lyrics/` (treat them as authoritative when present).
- Copy authoritative LRCs into `LYRICS_DB_FOLDER` using a sanitized canonical name.
- Remove non-audio files from `songs/` (keep .flac, .mp3, .wav).
"""

import os
import glob
import shutil
from typing import Optional
from .utils import sanitize_filename, extract_metadata_from_flac, parse_lrc_header_tags
from .logs_manager import dbg, info
from config import SONGS_FOLDER, LYRICS_FOLDER, LYRICS_DB_FOLDER

def ingest_existing_lrcs_and_cleanup_songs():
    """
    Move found .lrc files from songs/ to lyrics/ and clean non-audio files in songs/.
    """
    if not os.path.exists(SONGS_FOLDER):
        dbg("songs folder not present; skipping ingest")
        return

    info("Ingesting .lrc files found under songs/ and cleaning songs/...")
    lrc_patterns = [os.path.join(SONGS_FOLDER, "*.lrc"), os.path.join(SONGS_FOLDER, "*.LRC")]
    lrc_files = []
    for p in lrc_patterns:
        lrc_files.extend(glob.glob(p))

    for lrc_path in lrc_files:
        try:
            basename = os.path.basename(lrc_path)
            dest_lyrics = os.path.join(LYRICS_FOLDER, basename)

            # If destination exists, remove it to ensure authoritative replace.
            try:
                if os.path.exists(dest_lyrics):
                    os.remove(dest_lyrics)
                    dbg(f"existing lyrics file removed to be replaced: {dest_lyrics}")
            except Exception as e:
                dbg(f"could not remove existing lyrics file {dest_lyrics}: {e}")

            # Prefer move; fallback to copy+remove for cross-device/perm issues.
            try:
                shutil.move(lrc_path, dest_lyrics)
                dbg(f"moved {basename} to {LYRICS_FOLDER}")
            except Exception:
                try:
                    shutil.copy2(lrc_path, dest_lyrics)
                    os.remove(lrc_path)
                    dbg(f"copied then removed original (move fallback) {basename} to {LYRICS_FOLDER}")
                except Exception as e2:
                    info(f"warning: could not move or copy {basename} to {LYRICS_FOLDER}: {e2}")
                    continue  # skip further processing for this file

            # Determine canonical DB name using associated FLAC metadata or LRC tags.
            name_no_ext = os.path.splitext(basename)[0]
            corresponding_flac = os.path.join(SONGS_FOLDER, f"{name_no_ext}.flac")
            artist_meta = None
            title_meta = None
            if os.path.exists(corresponding_flac):
                artist_meta, title_meta = extract_metadata_from_flac(corresponding_flac)

            # If metadata missing, try LRC header tags now in dest_lyrics.
            if not (artist_meta and title_meta):
                lrc_artist, lrc_title = parse_lrc_header_tags(dest_lyrics)
                if lrc_artist and lrc_title:
                    if not artist_meta:
                        artist_meta = lrc_artist
                    if not title_meta:
                        title_meta = lrc_title

            if artist_meta and title_meta:
                canonical_db_name = sanitize_filename(f"{title_meta} - {artist_meta}")
            else:
                canonical_db_name = sanitize_filename(name_no_ext)

            db_dest = os.path.join(LYRICS_DB_FOLDER, f"{canonical_db_name}.lrc")

            # Copy authoritative LRC into DB, overwriting existing DB entry.
            try:
                if os.path.exists(db_dest):
                    os.remove(db_dest)
                    dbg(f"existing DB entry removed to be replaced: {db_dest}")
                shutil.copy2(dest_lyrics, db_dest)
                dbg(f"copied to DB {os.path.basename(db_dest)} (overwrote if existed)")
            except Exception as e:
                info(f"warning: error copying {dest_lyrics} to DB: {e}")
        except Exception as e:
            info(f"warning: error processing {lrc_path}: {e}")

    # Remove non-audio files from songs/ to keep folder tidy.
    info("Cleaning songs/: deleting non-audio files...")
    allowed_exts = {'.flac', '.mp3', '.wav'}
    for entry in os.listdir(SONGS_FOLDER):
        fpath = os.path.join(SONGS_FOLDER, entry)
        if os.path.isfile(fpath):
            ext = os.path.splitext(entry)[1].lower()
            if ext not in allowed_exts:
                try:
                    os.remove(fpath)
                    dbg(f"removed file {entry} from songs/")
                except Exception as e:
                    info(f"warning: could not remove {entry}: {e}")
    dbg("ingest and cleanup finished")
