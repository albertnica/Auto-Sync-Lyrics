"""
modules/lyrics_checker.py
Normalize ASR segment outputs and align them against authoritative lyrics text.

Responsibilities:
- Robustly parse and normalize different segment formats produced by ASR models:
  JSON objects with 'chunks', JSON lists, JSONL lines, or per-segment token lists.
- Convert segments into a canonical form: {'audio_path','start','end','text','tokens'}.
- Build flat token lists (with seg_idx) from either token-level timestamps or
  per-segment text (approximate per-word timing).
- Provide a global alignment engine (DP-based) to match lyric words to ASR tokens.
- Map segments to lyric lines, compute per-line timestamps, interpolate missing
  timestamps, and emit grouped JSONL and a CSV-like debug log.
- Maintain backward compatibility: public function names and behaviors are kept
  stable so `main.py` and other modules require no changes.
"""

import re
import json
from difflib import SequenceMatcher
from typing import List, Dict, Optional
from pathlib import Path

RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", flags=re.UNICODE)

def normalize(s: str) -> str:
    """
    Normalize whitespace and lowercase a string for comparisons.
    """
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def words_from_text(s: str) -> List[str]:
    """
    Extract word tokens from text using the shared RE_WORD regex.
    """
    return [m.group(0) for m in RE_WORD.finditer(s or "")]

def similarity(a: str, b: str) -> float:
    """
    Compute a similarity ratio between two strings using SequenceMatcher.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def read_segments_flex(path: Path) -> List[Dict]:
    """
    Read a flexible segment file (JSON object/list or JSONL) and normalize to segments.
    """
    txt = path.read_text(encoding="utf-8", errors="replace")
    obj = None
    try:
        obj = json.loads(txt)
    except Exception:
        obj = None
    segments = []
    if isinstance(obj, dict) and 'chunks' in obj and isinstance(obj['chunks'], list):
        for ch in obj['chunks']:
            seg = normalize_chunk_entry(ch, default_audio=path.stem)
            if seg:
                segments.append(seg)
        return segments
    if isinstance(obj, list):
        for e in obj:
            seg = normalize_chunk_entry(e, default_audio=path.stem)
            if seg:
                segments.append(seg)
        if segments:
            return segments
    # fallback: parse per-line JSON
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            j = json.loads(s)
        except Exception:
            try:
                j = eval(s, {}, {})  # trusted files assumed
            except Exception:
                continue
        seg = normalize_chunk_entry(j, default_audio=path.stem)
        if seg:
            segments.append(seg)
    return segments

def normalize_chunk_entry(ch, default_audio=None):
    """
    Normalize various chunk formats into a canonical segment dict or return None.
    """
    audio = ch.get('audio_path') if isinstance(ch, dict) else None
    start = None; end = None
    tokens = None
    # optional token-level timestamps
    if isinstance(ch, dict) and 'tokens' in ch:
        tokens = ch.get('tokens')
    # common timestamp forms
    if isinstance(ch, dict) and 'timestamp' in ch:
        t = ch['timestamp']
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            try:
                start = float(t[0]); end = float(t[1])
            except: start = None; end = None
    if isinstance(ch, dict) and ('start' in ch or 'end' in ch):
        try:
            if 'start' in ch: start = float(ch['start'])
            if 'end' in ch: end = float(ch['end'])
        except: pass
    # fallback: infer from token timestamps
    if (start is None or end is None) and tokens and isinstance(tokens, (list, tuple)) and len(tokens) > 0:
        try:
            first = tokens[0]
            last = tokens[-1]
            s0 = first.get('start') or first.get('timestamp') or None
            eN = last.get('end') or last.get('timestamp') or None
            start = float(s0) if s0 is not None else None
            end = float(eN) if eN is not None else None
        except Exception:
            start = start; end = end
    if start is None or end is None:
        # missing timestamps -> not usable for alignment
        return None
    if end < start:
        start, end = end, start
    text = ch.get('text', '') if isinstance(ch, dict) else ''
    if isinstance(text, (list, tuple)):
        text = " ".join(map(str, text))
    text = str(text).strip()
    if not audio:
        audio = f"songs/{(default_audio or 'unknown')}.flac"
    return {'audio_path': audio, 'start': start, 'end': end, 'text': text, 'tokens': tokens}

def build_token_list(segments: List[Dict]) -> List[Dict]:
    """
    Build a flat list of token dicts (text,start,end,seg_idx) from segments.
    """
    toks = []
    for s_idx, seg in enumerate(segments):
        seg_tokens = seg.get('tokens')
        if seg_tokens and isinstance(seg_tokens, list) and len(seg_tokens) > 0:
            for t in seg_tokens:
                w = t.get('text') or t.get('word') or ''
                s = None; e = None
                if isinstance(t.get('timestamp'), (list, tuple)) and len(t.get('timestamp')) >= 2:
                    s = t['timestamp'][0]; e = t['timestamp'][1]
                else:
                    s = t.get('start') or t.get('s') or None
                    e = t.get('end') or t.get('e') or None
                try:
                    s = float(s) if s is not None else None
                    e = float(e) if e is not None else None
                except:
                    s = None; e = None
                if s is None or e is None:
                    s = seg['start']; e = seg['end']
                toks.append({'text': str(w).strip(), 'start': float(s), 'end': float(e), 'seg_idx': s_idx})
        else:
            words = words_from_text(seg.get('text', ''))
            n = len(words) if words else 0
            if n == 0:
                continue
            seg_s = seg['start']; seg_e = seg['end']
            dur = seg_e - seg_s if seg_e > seg_s else 0.0
            for i, w in enumerate(words):
                if dur > 0:
                    s = seg_s + (i * dur / n)
                    e = seg_s + ((i+1) * dur / n)
                else:
                    s = seg_s; e = seg_e
                toks.append({'text': w, 'start': float(s), 'end': float(e), 'seg_idx': s_idx})
    return toks

def align_words(lyric_words: List[str], token_words: List[str]) -> List[Optional[int]]:
    """
    Align lyric words to token words using dynamic programming and return token indices.
    """
    A = [w.lower() for w in lyric_words]
    B = [w.lower() for w in token_words]
    la = len(A); lb = len(B)
    match_score = 2.0
    fuzzy_score = 1.0
    gap = -1.0
    dp = [[0.0] * (lb + 1) for _ in range(la + 1)]
    bt = [[None] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        dp[i][0] = dp[i-1][0] + gap
        bt[i][0] = ('up', i-1, 0)
    for j in range(1, lb + 1):
        dp[0][j] = dp[0][j-1] + gap
        bt[0][j] = ('left', 0, j-1)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            sub = match_score if A[i-1] == B[j-1] else (fuzzy_score if similarity(A[i-1], B[j-1]) >= 0.75 else -0.5)
            scores = [
                (dp[i-1][j-1] + sub, 'diag'),
                (dp[i-1][j] + gap, 'up'),
                (dp[i][j-1] + gap, 'left'),
            ]
            best, move = max(scores, key=lambda x: x[0])
            dp[i][j] = best
            if move == 'diag':
                bt[i][j] = ('diag', i-1, j-1)
            elif move == 'up':
                bt[i][j] = ('up', i-1, j)
            else:
                bt[i][j] = ('left', i, j-1)
    i, j = la, lb
    alignA = []
    alignB = []
    while i > 0 or j > 0:
        step = bt[i][j]
        if not step:
            break
        move, pi, pj = step[0], step[1], step[2]
        if move == 'diag':
            alignA.append(i-1)
            alignB.append(j-1)
            i, j = pi, pj
        elif move == 'up':
            alignA.append(i-1)
            alignB.append(None)
            i, j = pi, pj
        else:
            alignA.append(None)
            alignB.append(j-1)
            i, j = pi, pj
    alignA.reverse(); alignB.reverse()
    mapping = []
    ai = 0; bi = 0
    for k in range(len(alignA)):
        aidx = alignA[k]; bidx = alignB[k]
        if aidx is None and bidx is not None:
            bi += 1
            continue
        if aidx is not None and bidx is None:
            mapping.append(None)
            ai += 1
            continue
        if aidx is not None and bidx is not None:
            mapping.append(bidx)
            ai += 1; bi += 1
    while len(mapping) < la:
        mapping.append(None)
    return mapping

def assign_segments_to_lines(segments: List[Dict], lyrics_lines: List[str], non_empty_idxs: List[int],
                             max_window=3, threshold=0.5):
    """
    Assign ASR segments to lyric lines and return line assignments and per-segment mapping.
    """
    tokens = build_token_list(segments)
    token_words = [t['text'] for t in tokens]
    # prepare lyric words and the line index for each word
    lyric_words = []
    word_line_idx = []
    for li, line in enumerate(lyrics_lines):
        words = words_from_text(line)
        for w in words:
            lyric_words.append(w)
            word_line_idx.append(li)
    if not lyric_words or not token_words:
        seg_to_lines = [None] * len(segments)
        line_assignments = {i: [] for i in range(len(lyrics_lines))}
        return line_assignments, seg_to_lines

    mapping = align_words(lyric_words, token_words)
    token_to_seg = [t.get('seg_idx') for t in tokens]

    seg_to_lines = [None] * len(segments)
    line_assignments = {i: [] for i in range(len(lyrics_lines))}

    # build per-line matched token indices via mapping
    line_to_token_idxs: Dict[int, List[int]] = {i: [] for i in range(len(lyrics_lines))}
    for widx, t_idx in enumerate(mapping):
        if t_idx is None:
            continue
        if not (0 <= t_idx < len(tokens)):
            continue
        line_idx = word_line_idx[widx]
        line_to_token_idxs.setdefault(line_idx, []).append(t_idx)

    # for each segment, collect which lines have tokens belonging to that segment
    for s_idx in range(len(segments)):
        matched_lines = []
        matched_token_idxs = []
        for li, token_idxs in line_to_token_idxs.items():
            matched = [ti for ti in token_idxs if token_to_seg[ti] == s_idx]
            if matched:
                matched_lines.append(li)
                matched_token_idxs.extend(matched)
        if matched_lines:
            a = min(matched_lines); b = max(matched_lines)
            score = float(len(matched_token_idxs))
            seg_to_lines[s_idx] = (a, b, score)
            for li in range(a, b+1):
                if s_idx not in line_assignments.get(li, []):
                    line_assignments[li].append(s_idx)
        else:
            seg_to_lines[s_idx] = None

    for i in range(len(lyrics_lines)):
        line_assignments.setdefault(i, [])

    return line_assignments, seg_to_lines

def compute_line_timestamps(line_assignments: Dict[int, List[int]], segments: List[Dict], lyrics_lines: List[str]):
    """Return dict line_index -> {'start':..., 'end':..., 'n_segments':...}"""
    line_timestamps = {}
    for li in range(len(lyrics_lines)):
        seg_idxs = line_assignments.get(li, [])
        if not seg_idxs:
            line_timestamps[li] = {'start': None, 'end': None, 'n_segments': 0}
            continue
        starts = []
        ends = []
        for si in seg_idxs:
            if 0 <= si < len(segments):
                starts.append(segments[si]['start'])
                ends.append(segments[si]['end'])
        if starts and ends:
            line_timestamps[li] = {'start': min(starts), 'end': max(ends), 'n_segments': len(seg_idxs)}
        else:
            line_timestamps[li] = {'start': None, 'end': None, 'n_segments': 0}
    return line_timestamps

def interpolate_missing_timestamps(line_timestamps: Dict[int, Dict], lyrics_lines: List[str]) -> Dict[int, Dict]:
    """
    Estimate missing per-line timestamps by linear interpolation between neighbors.
    """
    n = len(lyrics_lines)
    for i in range(n):
        if line_timestamps[i]['start'] is None:
            prev = i-1
            while prev >= 0 and line_timestamps[prev]['start'] is None:
                prev -= 1
            nxt = i+1
            while nxt < n and line_timestamps[nxt]['start'] is None:
                nxt += 1
            if prev >= 0 and nxt < n:
                prev_end = line_timestamps[prev]['end']
                next_start = line_timestamps[nxt]['start']
                if prev_end is not None and next_start is not None and next_start > prev_end:
                    gap = next_start - prev_end
                    steps = (nxt - prev)
                    pos = i - prev
                    est_start = prev_end + gap * (pos-1)/steps
                    est_end   = prev_end + gap * pos/steps
                    line_timestamps[i]['start'] = est_start
                    line_timestamps[i]['end']   = est_end
    return line_timestamps

def fill_null_timestamps_with_neighbors(line_timestamps: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Fill leading/trailing null timestamps using the nearest non-null neighbor.
    """
    n = len(line_timestamps)
    non_null_idxs = [i for i in range(n) if line_timestamps[i]['start'] is not None]
    if not non_null_idxs:
        return line_timestamps
    first = non_null_idxs[0]
    last  = non_null_idxs[-1]
    for i in range(0, first):
        line_timestamps[i]['start'] = line_timestamps[first]['start']
        line_timestamps[i]['end']   = line_timestamps[first]['end']
        line_timestamps[i]['n_segments'] = 0
    for i in range(last+1, n):
        line_timestamps[i]['start'] = line_timestamps[last]['start']
        line_timestamps[i]['end']   = line_timestamps[last]['end']
        line_timestamps[i]['n_segments'] = 0
    i = first
    while i <= last:
        if line_timestamps[i]['start'] is None:
            gap_start = i
            j = i
            while j <= last and line_timestamps[j]['start'] is None:
                j += 1
            gap_end = j - 1
            prev_idx = gap_start - 1
            next_idx = j
            for k in range(gap_start, gap_end+1):
                d_prev = k - prev_idx if prev_idx >= 0 else float('inf')
                d_next = next_idx - k if next_idx < n else float('inf')
                if d_prev <= d_next:
                    line_timestamps[k]['start'] = line_timestamps[prev_idx]['start']
                    line_timestamps[k]['end']   = line_timestamps[prev_idx]['end']
                else:
                    line_timestamps[k]['start'] = line_timestamps[next_idx]['start']
                    line_timestamps[k]['end']   = line_timestamps[next_idx]['end']
                line_timestamps[k]['n_segments'] = 0
            i = j
        else:
            i += 1
    return line_timestamps

def group_lines_by_start_and_emit(lines: List[str], line_ts: Dict[int, Dict], segments_info: Dict[int, List[int]], audio_path: str, out_jsonl: Path, out_log: Path):
    """
    Group consecutive lyric lines with identical start timestamps and emit JSONL and CSV log.
    """
    n = len(lines)
    i = 0
    out_entries = []
    logs = []
    while i < n:
        start_i = line_ts[i]['start']
        j = i
        while j+1 < n and line_ts[j+1]['start'] == start_i:
            j += 1
        combined_parts = [lines[k].strip() for k in range(i, j+1) if lines[k].strip() != ""]
        combined_text = " ".join(combined_parts).strip()
        entry = {
            'audio_path': audio_path,
            'line_start_index': i,
            'line_end_index': j,
            'start': start_i,
            'text': combined_text,
            'n_lines': j - i + 1
        }
        out_entries.append(entry)
        assigned_segs = []
        for k in range(i, j+1):
            assigned = segments_info.get(k, [])
            assigned_segs.extend(assigned)
        assigned_segs = sorted(set(assigned_segs))
        logs.append({
            'line_start': i,
            'line_end': j,
            'start': start_i,
            'n_lines': j-i+1,
            'assigned_segments': ",".join(map(str,assigned_segs)),
            'text_preview': (combined_text[:200] + ("..." if len(combined_text)>200 else ""))
        })
        i = j + 1

    with out_jsonl.open('w', encoding='utf-8') as jf:
        for e in out_entries:
            out_obj = {
                'audio_path': e['audio_path'],
                'start': e['start'],
                'text': e['text'],
                'line_start_index': e['line_start_index'],
                'line_end_index': e['line_end_index'],
                'n_lines': e['n_lines']
            }
            jf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    with out_log.open('w', encoding='utf-8', newline='') as lf:
        import csv
        writer = csv.DictWriter(lf, fieldnames=['line_start','line_end','start','n_lines','assigned_segments','text_preview'])
        writer.writeheader()
        for r in logs:
            writer.writerow(r)

    return out_entries, logs

def load_lyrics_lines(path: Path):
    """
    Load lyric file lines and return (lines, non_empty_line_indices).
    """
    raw = path.read_text(encoding='utf-8', errors='replace').splitlines()
    lines = [ln.rstrip("\n") for ln in raw]
    non_empty_idxs = [i for i, ln in enumerate(lines) if len(ln.strip()) >= 1]
    return lines, non_empty_idxs
