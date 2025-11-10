"""
modules/utils.py
General helper functions frequently used.

Responsibilities:
- provide simple logging helpers (dbg, info, warn) used across the project.
- offer file and string utilities (credential loading, filename sanitization, text normalization).
- find audio files and format timestamps for LRC output.
- extract metadata from FLAC files and parse LRC header tags.
"""

import os
import re
import glob
import json
from typing import List, Optional, Tuple

try:
    from mutagen.flac import FLAC
except Exception:
    FLAC = None

from config import SONGS_FOLDER, CREDENTIALS_PATH
from .logs_manager import dbg, info

# File utilities
def load_credentials(path: str = CREDENTIALS_PATH) -> dict:
    """
    Load credentials from a JSON file and return a dict.
    """
    if not os.path.exists(path):
        info(f"credentials not found at {path}")
        raise FileNotFoundError(f"Credentials file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        info(f"error loading credentials: {e}")
        raise ValueError(f"Failed to read credentials file {path}: {e}")
    if not isinstance(data, dict):
        info("credentials.json parsed but is not an object/dict")
        raise ValueError(f"Credentials file {path} does not contain a JSON object")
    # debug: list top-level keys (without values)
    try:
        keys = list(data.keys())
        dbg(f"credentials.json keys: {keys}")
    except Exception:
        dbg("could not list credentials.json keys")
    dbg("credentials loaded")
    return data

def sanitize_filename(text: str) -> str:
    """
    Return a filesystem-safe filename by stripping illegal chars and trimming length.
    """
    if text is None:
        return ""
    text = re.sub(r'[<>:"/\\|?*]', '', str(text))
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 200:
        text = text[:200].strip()
    return text

def normalize_text(s: str) -> str:
    """
    Normalize text to lowercase and collapse non-word characters to single spaces.
    """
    s = (s or "").lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_flac_files(folder: str = SONGS_FOLDER) -> List[str]:
    """
    Return a sorted list of .flac files in the given folder.
    """
    pattern = os.path.join(folder, "*.flac")
    return sorted(glob.glob(pattern))

def format_lrc_timestamp(seconds: float) -> str:
    """
    Format seconds as an LRC timestamp 'MM:SS.ss'.
    """
    try:
        s = float(seconds)
    except Exception:
        s = 0.0
    minutes = int(s // 60)
    secs = s - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"

# Metadata extraction
def extract_metadata_from_flac(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract artist and title from FLAC tags when available, otherwise infer from filename.
    """
    if os.path.exists(path) and FLAC is not None:
        try:
            audio = FLAC(path)
            artist = audio.get("artist", [None])[0]
            title = audio.get("title", [None])[0]
            if artist and title:
                dbg(f"metadata from FLAC: artist='{artist}', title='{title}'")
                return artist, title
        except Exception:
            dbg("mutagen could not read FLAC tags (or mutagen not available)")
    base = os.path.splitext(os.path.basename(path))[0]
    if " - " in base:
        parts = base.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return None, base

def parse_lrc_header_tags(lrc_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse [ti:] and [ar:] header tags from an LRC file and return (artist, title).
    """
    artist = None
    title = None
    try:
        with open(lrc_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                m_ti = re.match(r'^\s*\[ti\s*:\s*(.+?)\s*\]\s*$', line, re.I)
                m_ar = re.match(r'^\s*\[ar\s*:\s*(.+?)\s*\]\s*$', line, re.I)
                if m_ti:
                    title = m_ti.group(1).strip()
                if m_ar:
                    artist = m_ar.group(1).strip()
                if artist and title:
                    break
    except Exception:
        dbg("failed to parse LRC header tags")
    return artist, title
