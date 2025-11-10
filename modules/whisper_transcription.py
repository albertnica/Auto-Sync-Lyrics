"""
modules/whisper_transcription.py
Helper utilities to run ASR transcription using configured models via the
Hugging Face `transformers` pipeline.

Responsibilities:
- wrap pipeline initialization and invocation with stderr filtering and safe fallbacks so callers do not need to handle noisy HF/torch messages.
- manage device selection (CPU vs CUDA) and provide best-effort diagnostics about device choice.
- expose simple APIs: `transcribe_with_model` (single-model run) and `transcribe_iter_models`.
"""

import os
import sys
import warnings
import logging
from typing import List, Dict, Any, Optional
from config import TRANSCRIBE_MODELS
from .logs_manager import dbg, info

try:
    from transformers import pipeline
    from transformers import logging as transformers_logging
except Exception:
    pipeline = None
    transformers_logging = None

try:
    import torch
except Exception:
    torch = None

# Reduce transformers telemetry and verbose logs when available.
# These environment variables and logger adjustments attempt to quiet HF/transformers
# so CLI output remains readable.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
    if transformers_logging is not None:
        transformers_logging.set_verbosity_error()
except Exception:
    # Do not fail if logging APIs are unavailable.
    pass

class _StderrFilter:
    """
    Filter specific stderr patterns emitted by HF transformers to reduce noise.
    """
    def __init__(self, orig):
        self._orig = orig
        # Patterns to suppress from stderr to avoid noisy HF warnings.
        self._patterns = [
            "Using custom `forced_decoder_ids`",
            "Transcription using a multilingual Whisper will default to language detection"
        ]

    def write(self, s):
        """
        Write filtered stderr output, skipping known noisy lines.
        """
        try:
            for part in s.splitlines(keepends=True):
                # Skip any lines that match noisy patterns; otherwise forward to original stderr.
                if any(p in part for p in self._patterns):
                    continue
                self._orig.write(part)
        except Exception:
            self._orig.write(s)

    def flush(self):
        """
        Flush the underlying stderr stream.
        """
        try:
            self._orig.flush()
        except Exception:
            pass

    def fileno(self):
        """
        Return the file descriptor of the underlying stderr, or 1 on failure.
        """
        try:
            return self._orig.fileno()
        except Exception:
            return 1

def _detect_device(requested: int) -> int:
    """
    Auto-detect device: prefer CUDA if available, otherwise fall back to CPU.
    """
    if torch is None:
        info("torch not available -> using CPU.")
        return -1
    try:
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            # Prefer the first CUDA device by default.
            try:
                count = torch.cuda.device_count()
                if count > 0:
                    info("Auto-detected CUDA available -> using cuda:0")
                    return 0
            except Exception:
                # If cuda APIs fail for some reason, fall back to CPU.
                dbg("torch.cuda APIs failed during device detection; falling back to CPU.")
                return -1
        info("CUDA not available -> using CPU.")
        return -1
    except Exception as e:
        dbg(f"_detect_device error: {e}")
        return -1

def _safe_pipeline_init(model_name: str, device: int, **kwargs):
    """
    Initialize a transformers ASR pipeline safely and return it, or None on failure.
    """
    if pipeline is None:
        dbg("transformers.pipeline unavailable.")
        return None
    orig_stderr = sys.stderr
    try:
        # Temporarily replace stderr to suppress known noisy messages during init.
        sys.stderr = _StderrFilter(orig_stderr)
        p = pipeline('automatic-speech-recognition', model=model_name, device=device, **kwargs)
        return p
    except Exception as e:
        dbg(f"pipeline init error for {model_name} on device {device}: {e}")
        return None
    finally:
        sys.stderr = orig_stderr

def _safe_pipeline_call(pipeline_obj, audio_path: str, **kwargs):
    """
    Invoke the pipeline on an audio file while filtering noisy stderr; return output or None.
    """
    if pipeline_obj is None:
        return None
    orig_stderr = sys.stderr
    try:
        sys.stderr = _StderrFilter(orig_stderr)
        out = pipeline_obj(audio_path, return_timestamps=True, **kwargs)
        return out
    except Exception as e:
        dbg(f"pipeline call error: {e}")
        return None
    finally:
        sys.stderr = orig_stderr

def transcribe_with_model(model_name: str, audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Transcribe `audio_path` with `model_name`. Returns {'model':..., 'result':...} or None.
    """
    # device selection is automatic (prefer CUDA if available, otherwise CPU)
    actual_device = _detect_device(None)
    p = _safe_pipeline_init(model_name, device=actual_device)
    if p is None:
        dbg(f"Pipeline init failed for {model_name}.")
        return None
    out = _safe_pipeline_call(p, audio_path)
    if out is None:
        dbg(f"Pipeline returned no output for {model_name}.")
        return None
    # Return raw result
    return {"model": model_name, "result": out}

def transcribe_iter_models(audio_path: str, run_folder: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Try each configured model in order and collect successful transcription results.
    """
    results = []
    for m in TRANSCRIBE_MODELS:
        info(f"Transcribing with model: {m}")
        res = transcribe_with_model(m, audio_path)
        if res:
            results.append(res)
            # Not saving full result as .json to keep pipeline IO responsibility in caller.
            dbg(f"Model {m} succeeded.")
        else:
            dbg(f"Model {m} failed.")
    return results
