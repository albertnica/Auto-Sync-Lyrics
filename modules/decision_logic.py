"""
decision_logic.py
Scoring and selection logic to choose the best transcription model.

Responsibilities:
- Compare words from reference lyrics (TXT) and model JSONL output.
- Compute a simple score: coincidentes - jsonl_only * mult - txt_only.
- Provide a fallback selection using repeat_occurrences if no reference lyrics are available.
- Produce a debug log that summarizes counts and the selected metric components.
"""

import os
import json
import re
from typing import List, Dict, Optional
from config import TRANSCRIBE_MODELS

# Word extraction (supports accented letters and apostrophes)
RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", flags=re.UNICODE)


def extract_words(text: str) -> List[str]:
    """
    Extract lowercase word tokens from a text string.
    """
    return [m.group(0).lower() for m in RE_WORD.finditer(text or "")]


def read_txt_words(txt_path: Optional[str]) -> tuple:
    """
    Read a TXT lyrics file and return (words, lines); returns empty on failure.
    """
    if not txt_path:
        return [], []
    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        content = "".join(lines)
        words = extract_words(content)
        return words, lines
    except Exception:
        return [], []


def read_jsonl_words(jsonl_path: str) -> tuple:
    """
    Read a JSONL transcription file and return (words, objects).
    """
    words = []
    objects = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    objects.append(obj)
                    text = obj.get("text", "") or obj.get("words", "") or ""
                    words.extend(extract_words(text))
                except Exception:
                    continue
    except Exception:
        pass
    return words, objects


def count_lines_with_many_repetitions(jsonl_objects: List[dict]) -> tuple:
    """
    Detect lines with excessive repeated tokens and return (multiplier, details).
    """
    try:
        mult = 1
        lines_with_reps = {}

        for obj in jsonl_objects:
            text = obj.get("text", "").strip()
            if not text:
                continue

            words = extract_words(text)
            if not words:
                continue

            word_count = {}
            for w in words:
                word_count[w] = word_count.get(w, 0) + 1

            max_reps = max(word_count.values()) if word_count else 0
            # Legacy threshold preserved: if a line has extreme repetition, increment multiplier.
            if max_reps > 100:
                mult += 1
                preview = text[:100] + "..." if len(text) > 100 else text
                lines_with_reps[preview] = max_reps

        return mult, lines_with_reps
    except Exception:
        return 1, {}


def compute_repeat_occurrences(jsonl_words: List[str]) -> int:
    """
    Count total extra repeat occurrences across JSONL words.
    """
    cnt = {}
    for w in jsonl_words:
        cnt[w] = cnt.get(w, 0) + 1
    repeats = sum((c - 1) for c in cnt.values() if c > 1)
    return repeats


def write_debug_log(log_path: str, txt_lines: List[str], jsonl_objects: List[dict],
                   txt_words: List[str], jsonl_words: List[str],
                   coinc: int, txt_only: int, jsonl_only: int,
                   mult: int, rep_details: dict, score: float, repeat_occurrences: int):
    """
    Write a detailed debug log summarizing the scoring inputs and results.
    """
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("DEBUG LOG - DECISION LOGIC\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"REFERENCE TXT ({len(txt_lines)} lines):\n")
            f.write("-" * 80 + "\n")
            for i, line in enumerate(txt_lines, 1):
                f.write(f"{i:3d}: {line}")
            f.write("\n")

            f.write(f"JSONL FROM TRANSCRIBER ({len(jsonl_objects)} objects):\n")
            f.write("-" * 80 + "\n")
            for i, obj in enumerate(jsonl_objects, 1):
                timestamp = obj.get("timestamp", obj.get("start", "?"))
                text = obj.get("text", "")
                preview = text[:200] + "..." if len(text) > 200 else text
                f.write(f"{i:3d}: [{timestamp}] {preview}\n")
            f.write("\n")

            f.write(f"ALL WORDS IN TXT ({len(txt_words)} total):\n")
            f.write("-" * 80 + "\n")
            txt_count = {}
            for w in txt_words:
                txt_count[w] = txt_count.get(w, 0) + 1
            for w in sorted(txt_count.keys()):
                f.write(f"  {w}: {txt_count[w]}x\n")
            f.write("\n")

            f.write(f"ALL WORDS IN JSONL ({len(jsonl_words)} total):\n")
            f.write("-" * 80 + "\n")
            jsonl_count = {}
            for w in jsonl_words:
                jsonl_count[w] = jsonl_count.get(w, 0) + 1
            for w in sorted(jsonl_count.keys()):
                f.write(f"  {w}: {jsonl_count[w]}x\n")
            f.write("\n")

            num_lineas_rep = len(rep_details)
            f.write(f"LINES WITH VERY HIGH REPETITION (>100) ({num_lineas_rep}):\n")
            f.write("-" * 80 + "\n")
            if rep_details:
                for txt, max_count in sorted(rep_details.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  [max_word={max_count}x] {txt}\n")
            else:
                f.write("  None\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("FINAL RESULTS:\n")
            f.write("=" * 80 + "\n")
            f.write(f"Coincidentes: {coinc}\n")
            f.write(f"TXT only: {txt_only}\n")
            f.write(f"JSONL only: {jsonl_only}\n")
            f.write(f"Lines with repetition >100: {num_lineas_rep}\n")
            f.write(f"Multiplier (1 + repeated_lines): {mult}\n")
            f.write(f"Formula: score = {coinc} - {jsonl_only} * {mult} - {txt_only}\n")
            f.write(f"         score = {coinc} - {jsonl_only * mult} - {txt_only}\n")
            f.write(f"Final score: {score:.3f}\n")
            f.write(f"Repeat occurrences (total extra repeats in JSONL): {repeat_occurrences}\n")
    except Exception:
        pass


def compute_model_score(txt_words: List[str], jsonl_words: List[str],
                       txt_lines: List[str], jsonl_objects: List[dict],
                       log_path: str) -> Dict:
    """
    Compute scoring metrics comparing TXT words to JSONL words and return metrics dict.
    """
    jsonl_remaining = jsonl_words[:]

    coinc = 0
    txt_only = 0

    # For each TXT word, try to remove one match from JSONL remaining
    for word in txt_words:
        if word in jsonl_remaining:
            coinc += 1
            jsonl_remaining.remove(word)
        else:
            txt_only += 1

    # leftovers in JSONL
    jsonl_only = len(jsonl_remaining)

    # multiplier based on extremely repeated lines
    mult, rep_details = count_lines_with_many_repetitions(jsonl_objects)

    # final score
    score = coinc - (jsonl_only * mult) - txt_only

    repeat_occurrences = compute_repeat_occurrences(jsonl_words)

    try:
        write_debug_log(log_path, txt_lines, jsonl_objects, txt_words, jsonl_words,
                        coinc, txt_only, jsonl_only, mult, rep_details, score, repeat_occurrences)
    except Exception:
        pass

    return {
        "coincidentes": coinc,
        "txt_only": txt_only,
        "jsonl_only": jsonl_only,
        "lineas_repetidas_mas_threshold": len(rep_details),
        "mult": mult,
        "score": score,
        "repeat_occurrences": repeat_occurrences
    }


def compute_scores(model_evals: List[Dict], lyrics_txt_path: Optional[str]) -> List[Dict]:
    """
    Compute metrics for multiple model evaluations given an optional TXT reference.
    """
    txt_words, txt_lines = read_txt_words(lyrics_txt_path)

    enriched = []
    for ev in model_evals:
        model_name = ev.get("model")
        raw_jsonl_path = ev.get("raw_jsonl")
        aligned_jsonl_path = ev.get("aligned_jsonl")

        result = {
            "model": model_name,
            "raw_jsonl": raw_jsonl_path,
            "aligned_jsonl": aligned_jsonl_path,
            "entries_count": ev.get("entries_count", 0),
            "coincidentes": 0,
            "txt_only": 0,
            "jsonl_only": 0,
            "lineas_repetidas_mas_threshold": 0,
            "mult": 1,
            "score": 0.0,
            "repeat_occurrences": 0
        }

        if raw_jsonl_path and os.path.exists(raw_jsonl_path):
            jsonl_words, jsonl_objects = read_jsonl_words(raw_jsonl_path)

            jsonl_dir = os.path.dirname(raw_jsonl_path)
            jsonl_base = os.path.basename(raw_jsonl_path)
            log_name = jsonl_base.replace(".jsonl", "_debug.log")
            log_path = os.path.join(jsonl_dir, log_name)

            metrics = compute_model_score(txt_words, jsonl_words, txt_lines, jsonl_objects, log_path)
            result.update(metrics)

        enriched.append(result)

    return enriched


def choose_best_model(enriched: List[Dict], fallback_to_transcription: bool = False) -> Optional[Dict]:
    """
    Select the best model from enriched metrics; use score or repeat_occurrences.
    """
    if not enriched:
        return None

    if fallback_to_transcription:
        min_rep = min((e.get('repeat_occurrences', 0) for e in enriched))
        candidates = [e for e in enriched if e.get('repeat_occurrences', 0) == min_rep]
        if len(candidates) == 1:
            best = candidates[0]
        else:
            order_map = {m: i for i, m in enumerate(TRANSCRIBE_MODELS)}
            candidates.sort(key=lambda x: order_map.get(x.get('model'), 9999))
            best = candidates[0]
        best["_decision_score"] = float(-best.get('repeat_occurrences', 0))
        return best

    max_score = max((e.get('score', float('-inf')) for e in enriched))
    candidates = [e for e in enriched if e.get('score', float('-inf')) == max_score]
    if not candidates:
        return None
    if len(candidates) == 1:
        best = candidates[0]
    else:
        order_map = {m: i for i, m in enumerate(TRANSCRIBE_MODELS)}
        candidates.sort(key=lambda x: order_map.get(x.get('model'), 9999))
        best = candidates[0]

    best["_decision_score"] = float(best.get('score', 0.0))
    return best
