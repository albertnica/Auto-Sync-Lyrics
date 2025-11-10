"""
db_manager.py
Minimal management of the lyrics output folder and the normalized lyrics DB.

Responsibilities:
- Ensure required folders exist.
- Initialize and clean output folders.
- Copy generated LRC files into the normalized DB using canonical "<Title> - <Artist>.lrc" names.
- Restore or search for exact/case-insensitive DB entries.
"""

import os
import glob
import shutil
from typing import Optional
from .utils import sanitize_filename
from .logs_manager import dbg, info
from config import LYRICS_FOLDER, LYRICS_DB_FOLDER


def ensure_folders():
    """
    Ensure the lyrics output and DB folders exist.
    """
    os.makedirs(LYRICS_FOLDER, exist_ok=True)
    os.makedirs(LYRICS_DB_FOLDER, exist_ok=True)


def initialize_folders():
    """
    Initialize (recreate) lyrics and DB folders, removing existing contents.
    """
    info("Initializing folders...")
    if os.path.exists(LYRICS_FOLDER):
        info(f"  Removing existing '{LYRICS_FOLDER}' directory...")
        shutil.rmtree(LYRICS_FOLDER)
    os.makedirs(LYRICS_FOLDER, exist_ok=True)
    os.makedirs(LYRICS_DB_FOLDER, exist_ok=True)
    dbg(f"created folders: {LYRICS_FOLDER}, {LYRICS_DB_FOLDER}")


def copy_to_lyrics_db(lrc_path: str, title: str, artist: str):
    """
    Copy an LRC file into the canonical lyrics DB under a sanitized name.
    """
    try:
        normalized_name = sanitize_filename(f"{title} - {artist}")
        db_filename = f"{normalized_name}.lrc"
        db_path = os.path.join(LYRICS_DB_FOLDER, db_filename)
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
                dbg(f"removed existing DB entry to overwrite: {db_filename}")
            shutil.copy2(lrc_path, db_path)
            dbg(f"copied to DB: {db_filename}")
        except Exception as e:
            info(f"warning: error copying to DB: {e}")
    except Exception as e:
        info(f"warning: error copying to DB: {e}")


def restore_from_lyrics_db(title: str, artist: str, target_lrc_path: str) -> bool:
    """
    Restore a canonical LRC from the DB to a target path if it exists.
    """
    try:
        normalized_name = sanitize_filename(f"{title} - {artist}")
        db_filename = f"{normalized_name}.lrc"
        db_path = os.path.join(LYRICS_DB_FOLDER, db_filename)
        if os.path.exists(db_path):
            shutil.copy2(db_path, target_lrc_path)
            info(f"Restored from DB: {db_filename}")
            return True
        else:
            dbg(f"not found in DB: {db_filename}")
            return False
    except Exception as e:
        info(f"warning: error restoring from DB: {e}")
        return False


def search_similar_in_db(title: str, artist: str) -> Optional[str]:
    """
    Search for an exact or case-insensitive matching LRC in the DB and return its path.
    """
    try:
        if not os.path.exists(LYRICS_DB_FOLDER):
            return None
        target_norm = sanitize_filename(f"{title} - {artist}")
        target_filename = f"{target_norm}.lrc"
        target_path = os.path.join(LYRICS_DB_FOLDER, target_filename)
        if os.path.exists(target_path):
            info(f"Found exact DB entry: {target_filename}")
            return target_path
        # case-insensitive scan
        db_files = glob.glob(os.path.join(LYRICS_DB_FOLDER, "*.lrc"))
        target_lower = target_norm.lower()
        for f in db_files:
            base = os.path.splitext(os.path.basename(f))[0]
            if base.lower() == target_lower:
                info(f"Found DB entry by case-insensitive match: {os.path.basename(f)}")
                return f
        return None
    except Exception as e:
        info(f"warning: error searching DB: {e}")
        return None
