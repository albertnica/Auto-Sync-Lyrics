"""
modules/lrc_writer.py

Responsibilities:
- JSONL → LRC writer that accepts common timestamp shapes found in JSONL
  (e.g. 'start', 'timestamp' arrays or dicts, 'begin', 'time'), normalizes and validates
  numeric start times, and preserves multiline text content.
"""

import json
import os
from typing import List, Dict, Any, Optional
from .utils import format_lrc_timestamp
from .logs_manager import dbg

def _extract_start_from_obj(obj: Dict[str, Any]) -> Optional[float]:
    """
    Extract a numeric start time from a flexible object shape, or return None.
    """
    cand = None
    if 'start' in obj and obj['start'] is not None:
        cand = obj['start']
    elif 'timestamp' in obj and obj['timestamp'] is not None:
        t = obj['timestamp']
        if isinstance(t, (list, tuple)) and len(t) >= 1:
            cand = t[0]
        elif isinstance(t, dict):
            if 'start' in t and t['start'] is not None:
                cand = t['start']
            elif 'begin' in t and t['begin'] is not None:
                cand = t['begin']
            elif 'time' in t and t['time'] is not None:
                cand = t['time']
        else:
            cand = t
    elif 'begin' in obj and obj['begin'] is not None:
        cand = obj['begin']
    elif 'time' in obj and obj['time'] is not None:
        cand = obj['time']

    if cand is None:
        return None
    try:
        if isinstance(cand, (list, tuple)) and len(cand) >= 1:
            cand_val = cand[0]
        else:
            cand_val = cand
        if isinstance(cand_val, dict):
            if 'start' in cand_val:
                cand_val = cand_val['start']
            elif 'begin' in cand_val:
                cand_val = cand_val['begin']
            else:
                return None
        return float(cand_val)
    except Exception:
        return None

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return a list of normalized items with 'start' and 'text'.
    """
    items: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        dbg(f"read_jsonl: input not found {path}")
        return items

    discarded = 0
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                dbg(f"read_jsonl: line {lineno} JSON error -> {e}")
                continue

            start = _extract_start_from_obj(obj)
            if start is None:
                dbg(f"read_jsonl: line {lineno} missing/invalid start -> discarded")
                discarded += 1
                continue

            text = obj.get("text")
            if text is None:
                text = obj.get("words") or obj.get("transcript") or ""
            try:
                text = str(text)
            except Exception:
                text = ""

            items.append({"start": float(start), "text": text})

    if discarded:
        dbg(f"read_jsonl: discarded {discarded} lines without valid start from {path}")
    return items

def write_lrc(entries: List[Dict[str, Any]], out_path: str):
    """
    Write a list of entries to an LRC file, normalizing timestamps and content.
    """
    # Normalize and filter entries: keep only those with a valid numeric start
    valid_entries: List[Dict[str, Any]] = []
    invalid_count = 0
    for e in entries:
        try:
            s = e.get("start")
            if s is None:
                invalid_count += 1
                continue
            s_f = float(s)
            if s_f < 0:
                s_f = 0.0
            valid_entries.append({"start": s_f, "text": e.get("text", "")})
        except Exception:
            invalid_count += 1

    if invalid_count:
        dbg(f"write_lrc: skipped {invalid_count} entries without valid numeric start")

    # Sort by start (all entries here have a valid 'start' numeric)
    entries_sorted = sorted(valid_entries, key=lambda x: x["start"])

    # Ensure parent folder exists
    parent = os.path.dirname(out_path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception:
            pass

    # Write LRC: each sub-line receives the same timestamp
    with open(out_path, "w", encoding="utf-8", newline="\n") as out_f:
        for e in entries_sorted:
            text = e.get("text", "")
            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception:
                    text = ""
            if text.strip() == "":
                continue
            start_f = e.get("start")
            timestamp = format_lrc_timestamp(start_f)
            lines = text.splitlines() or [text]
            for sub in lines:
                sub = sub.strip()
                if sub:
                    out_f.write(f"[{timestamp}]{sub}\n")
