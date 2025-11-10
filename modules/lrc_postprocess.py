"""
modules/lrc_postprocess.py
LRC post-processing utilities.

Responsibilities:
 - The first LRC line is preserved untouched.
 - Lines may be split at tokens that start with an uppercase letter,
   unless excluded by preceding characters or exception lists.
 - proper_names.txt provides additional exception tokens.
"""

from typing import List, Tuple, Optional, Set
import re
import unicodedata
import os
from modules import utils as mutils

# Token regex
_RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", flags=re.UNICODE)
# Parse LRC lines: [MM:SS.ss]text
_RE_LRC = re.compile(r'^\s*\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$')

# Base tokens that should NOT trigger a split (lowercase)
_EXCEPT_TOKENS_BASE = {
    "i", "i'm", "i've", "i'll", "i'd",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "january","february","march","april","may","june","july","august","september","october","november","december",
}

# Characters that, if immediately preceding a token, prevent a split (e.g. "(Oh")
_PRECEDING_EXCLUDE_CHARS = set(['(', '[', '{', '"', "'", "¡", "¿"])

# File with proper names (one per line) at project root
_PROPER_NAMES_FILENAME = "proper_names.txt"

def _load_proper_names(path: str) -> Set[str]:
    """
    Load proper names from a file (one per line) used as exceptions for splitting.
    """
    names = set()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    names.add(line.lower())
        else:
            mutils.dbg(f"lrc_postprocess: proper_names.txt not found at {path}; continuing without it.")
    except Exception as e:
        mutils.dbg(f"lrc_postprocess: error loading proper names from {path}: {e}")
    return names

def _parse_lrc(path: str) -> List[Tuple[float, str]]:
    """
    Parse an LRC file into a list of (start_seconds, text) tuples.
    """
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                m = _RE_LRC.match(raw)
                if not m:
                    continue
                minutes = int(m.group(1))
                seconds = float(m.group(2))
                start = minutes * 60.0 + seconds
                text = m.group(3).rstrip("\n")
                out.append((start, text))
    except Exception as e:
        mutils.dbg(f"lrc_postprocess._parse_lrc: error reading {path}: {e}")
    return out

def _format_lrc_ts(seconds: float) -> str:
    """
    Format seconds as MM:SS.ss for LRC output.
    """
    if seconds < 0:
        seconds = 0.0
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"

def _is_uppercase_start(ch: str) -> bool:
    """
    Return True when the character has an uppercase Unicode category.
    """
    if not ch:
        return False
    return unicodedata.category(ch).startswith("Lu")

def _token_spans(text: str) -> List[Tuple[int,int,str]]:
    """
    Return spans (start,end,token) for word tokens in text.
    """
    out = []
    for m in _RE_WORD.finditer(text):
        out.append((m.start(), m.end(), m.group(0)))
    return out

def _fragments_from_spans_and_splits(text: str, spans: List[Tuple[int,int,str]], split_token_indices: List[int]) -> List[str]:
    """
    Build text fragments from token spans and chosen split indices.
    """
    if not spans:
        return [text.strip()]

    last_pos = 0
    fragments = []
    for sidx in split_token_indices:
        start_char = spans[sidx][0]
        frag = text[last_pos:start_char].strip()
        if frag:
            fragments.append(frag)
        last_pos = start_char
    tail = text[last_pos:].strip()
    if tail:
        fragments.append(tail)
    if not fragments:
        return [text.strip()]
    return fragments

def split_capitalized_intralines(lrc_path: str, out_path: Optional[str] = None, fallback_dur: float = 5.0, min_piece_dur: float = 0.25):
    """
    Split LRC lines at capitalized tokens when appropriate and write output.
    """
    entries = _parse_lrc(lrc_path)
    if not entries:
        mutils.dbg("lrc_postprocess: no LRC entries parsed.")
        if out_path and out_path != lrc_path:
            with open(lrc_path, "r", encoding="utf-8", errors="replace") as fr, open(out_path, "w", encoding="utf-8") as fw:
                fw.write(fr.read())
        return

    # Load proper names and compose exception list
    proper_names = _load_proper_names(_PROPER_NAMES_FILENAME)
    EXCEPT_TOKENS = set(_EXCEPT_TOKENS_BASE)
    EXCEPT_TOKENS.update(proper_names)

    starts = [s for s, _ in entries]
    texts = [t for _, t in entries]
    n = len(entries)
    result: List[Tuple[float, str]] = []

    for i in range(n):
        start = starts[i]
        end = starts[i+1] if i+1 < n else (start + fallback_dur)
        text = texts[i]

        # Rule: never split the first file line
        if i == 0:
            result.append((start, text.strip()))
            continue

        spans = _token_spans(text)
        if not spans:
            result.append((start, text.strip()))
            continue

        # Choose split token indices: token index >0, uppercase start, not in exceptions, not preceded by excluded chars
        split_token_indices: List[int] = []
        for idx, (s_char, e_char, tok) in enumerate(spans):
            if idx == 0:
                # keep first token intact
                continue
            first_char = tok[0]
            if not _is_uppercase_start(first_char):
                continue
            tok_l = tok.lower()
            if tok_l in EXCEPT_TOKENS:
                continue
            prev_idx = s_char - 1
            if prev_idx >= 0:
                prev_ch = text[prev_idx]
                if prev_ch in _PRECEDING_EXCLUDE_CHARS:
                    continue
            split_token_indices.append(idx)

        if not split_token_indices:
            result.append((start, text.strip()))
            continue

        fragments = _fragments_from_spans_and_splits(text, spans, split_token_indices)
        if len(fragments) <= 1:
            result.append((start, text.strip()))
            continue

        # distribute duration proportionally by fragment character counts
        lens = [len(f) for f in fragments]
        total_chars = sum(lens) if sum(lens) > 0 else len(fragments)
        durations = [ (end - start) * (l / total_chars) for l in lens ]

        # merge pieces shorter than min_piece_dur with previous fragment
        merged_frags: List[str] = []
        merged_durs: List[float] = []
        for frag, dur in zip(fragments, durations):
            if merged_frags and dur < min_piece_dur:
                merged_frags[-1] = merged_frags[-1] + " " + frag
                merged_durs[-1] += dur
            else:
                merged_frags.append(frag)
                merged_durs.append(dur)

        # append fragments with recalculated starts
        cur = start
        result_count_before = len(result)
        for frag, dur in zip(merged_frags, merged_durs):
            result.append((cur, frag.strip()))
            cur += dur
        mutils.dbg(f"lrc_postprocess: split at {start:.2f}s (line {i}): {len(merged_frags)} parts (orig {len(fragments)}), wrote {len(result)-result_count_before} entries")

    # sort and write output
    result.sort(key=lambda x: x[0])
    if out_path is None:
        out_path = lrc_path

    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as fw:
            for s, txt in result:
                txt_trim = (txt or "").strip()
                if not txt_trim:
                    continue
                ts = _format_lrc_ts(s)
                fw.write(f"[{ts}]{txt_trim}\n")
        mutils.dbg(f"lrc_postprocess: wrote postprocessed LRC to {out_path}")
    except Exception as e:
        mutils.dbg(f"lrc_postprocess: error writing {out_path}: {e}")

def append_trailing_empty_line(lrc_path: str, extra_seconds: float = 20.0, out_path: Optional[str] = None):
    """
    Append an empty timestamp line sufficiently after the last LRC timestamp.
    """
    entries = _parse_lrc(lrc_path)
    if not entries:
        mutils.dbg("lrc_postprocess.append_trailing_empty_line: no entries parsed; nothing appended.")
        if out_path and out_path != lrc_path:
            with open(lrc_path, "r", encoding="utf-8", errors="replace") as fr, open(out_path, "w", encoding="utf-8") as fw:
                fw.write(fr.read())
        return

    last_start = max(s for s, _ in entries)
    new_ts = last_start + float(extra_seconds)
    new_line = f"[{_format_lrc_ts(new_ts)}]\n"

    if out_path is None:
        out_path = lrc_path

    try:
        with open(lrc_path, "r", encoding="utf-8", errors="replace") as fr:
            original = fr.read()
    except Exception as e:
        mutils.dbg(f"lrc_postprocess.append_trailing_empty_line: failed to read {lrc_path}: {e}")
        return

    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as fw:
            fw.write(original.rstrip("\n") + "\n" + new_line)
        mutils.dbg(f"lrc_postprocess: appended trailing empty timestamp { _format_lrc_ts(new_ts) } to {out_path}")
    except Exception as e:
        mutils.dbg(f"lrc_postprocess.append_trailing_empty_line: failed to write {out_path}: {e}")
