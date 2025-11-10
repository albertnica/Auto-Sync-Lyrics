"""
main.py
Primary orchestrator.

Responsibilities:
 - create a per-run log folder and initialize run log
 - prepare output and DB folders and ingest any existing .lrc files next to songs/
 - iterate .flac files in SONGS_FOLDER:
     * attempt exact restore from lyrics DB
     * try to retrieve authoritative lyrics from Genius (stored on .logs)
     * transcribe audio with multiple ASR models (in preference order)
     * produce JSONL per-model and, when a lyrics txt is available, align segments -> per-line timestamps
     * compute model scores with decision_logic and select the best
     * write final .lrc using chosen aligned JSONL
     * postprocess LRC and copy canonical entry into lyrics DB
"""

import os
import shutil
from pathlib import Path
from modules import utils as mutils
from modules import logs_manager, db_manager, ingest, lyrics_retrieving, whisper_transcription, lyrics_checker, lrc_writer, decision_logic
from modules.logs_manager import init_anchors_log, append_anchor_entry, make_run_folder
from config import LYRICS_FOLDER, TRANSCRIBE_MODELS, FALLBACK_TO_TRANSCRIPTION_ON_MISSING_LYRICS
import json

def process_all():
    # Create run folder and initialize main log
    run_folder = make_run_folder()
    init_anchors_log(run_folder)

    # Initialize folders (this recreates lyrics/ clean)
    db_manager.initialize_folders()
    # Ingest existing LRCs and clean songs/
    ingest.ingest_existing_lrcs_and_cleanup_songs()

    flacs = mutils.find_flac_files()
    if not flacs:
        print("No .flac files found in songs/. Nothing to do.")
        return

    for fpath in flacs:
        basename = os.path.basename(fpath)
        flac_base = os.path.splitext(basename)[0]
        print(f"Processing {basename} ...")

        # Extract metadata
        artist, title = mutils.extract_metadata_from_flac(fpath)

        # Attempt exact DB restore (no writing to DB here) -> if found, copy to lyrics/<flac_base>.lrc and continue
        restored = False
        if artist and title:
            try:
                db_entry = db_manager.search_similar_in_db(title, artist)
                if db_entry:
                    dest = os.path.join(LYRICS_FOLDER, f"{flac_base}.lrc")
                    try:
                        os.makedirs(LYRICS_FOLDER, exist_ok=True)
                        shutil.copy2(db_entry, dest)
                        append_anchor_entry(run_folder, basename, "restored_from_db_mainloop", os.path.basename(db_entry))
                        print(f"Restored lyrics from DB for {basename} as {os.path.basename(dest)}")
                        restored = True
                    except Exception as e:
                        mutils.dbg(f"Could not copy DB entry to lyrics/: {e}")
            except Exception as e:
                mutils.dbg(f"DB restore attempt failed: {e}")
        if restored:
            continue

        # If no DB restore, attempt Genius retrieval (returns path to run_folder audit .txt)
        lyrics_txt_path = lyrics_retrieving.retrieve_for_flac(fpath, run_folder=run_folder)
        if not lyrics_txt_path:
            # If configured to fallback, continue; otherwise skip file
            if FALLBACK_TO_TRANSCRIPTION_ON_MISSING_LYRICS:
                mutils.dbg(f"No lyrics for {basename} but FALLBACK_TO_TRANSCRIPTION_ON_MISSING_LYRICS is True -> continuing with transcription-only flow.")
                # lyrics_txt_path remains None to indicate fallback-to-transcription mode
            else:
                append_anchor_entry(run_folder, basename, "lyrics_retrieval_failed", "")
                print(f"No lyrics obtained for {basename}; skipping transcription/align.")
                continue

        # Transcribe with multiple models
        trans_results = whisper_transcription.transcribe_iter_models(fpath, run_folder=run_folder)
        if not trans_results:
            append_anchor_entry(run_folder, basename, "all_models_failed", "")
            print(f"All transcription models failed for {basename}")
            continue

        # For each model: persist segments, optionally run lyrics_checker alignment,
        # Otherwise, in fallback mode we keep raw_jsonl as aligned_jsonl so a later LRC can be built from it
        model_evals = []
        for tr in trans_results:
            model_name = tr.get("model")
            out = tr.get("result")
            fname = flac_base
            safe_model_tag = model_name.replace("/", "_")
            tmp_jsonl = os.path.join(run_folder, f"{fname}__{safe_model_tag}.jsonl")
            try:
                # Persist segments/chunks depending on pipeline shape
                segs = []
                if isinstance(out, dict):
                    if 'chunks' in out and isinstance(out['chunks'], list):
                        segs = out['chunks']
                    elif 'segments' in out and isinstance(out['segments'], list):
                        segs = out['segments']
                    else:
                        segs = [{'start': 0.0, 'end': 9999.0, 'text': out.get('text', '')}]
                elif isinstance(out, list):
                    segs = out
                else:
                    segs = [{'start': 0.0, 'end': 9999.0, 'text': str(out)}]

                # Save segments as JSONL for checker input
                with open(tmp_jsonl, "w", encoding="utf-8") as fh:
                    for s in segs:
                        fh.write(json.dumps(s, ensure_ascii=False) + "\n")

                # If we have authoritative lyrics, run alignment pipeline
                if lyrics_txt_path:
                    segments = lyrics_checker.read_segments_flex(Path(tmp_jsonl))
                    lyrics_lines, non_empty_idxs = lyrics_checker.load_lyrics_lines(Path(lyrics_txt_path))
                    line_assignments, seg_to_lines = lyrics_checker.assign_segments_to_lines(
                        segments, lyrics_lines, non_empty_idxs, max_window=3, threshold=0.50)
                    line_ts = lyrics_checker.compute_line_timestamps(line_assignments, segments, lyrics_lines)
                    line_ts = lyrics_checker.interpolate_missing_timestamps(line_ts, lyrics_lines)
                    line_ts = lyrics_checker.fill_null_timestamps_with_neighbors(line_ts)

                    out_jsonl = os.path.join(run_folder, f"{fname}__{safe_model_tag}_aligned.jsonl")
                    out_log = os.path.join(run_folder, f"{fname}__{safe_model_tag}_aligned_log.csv")
                    entries, logs = lyrics_checker.group_lines_by_start_and_emit(
                        lyrics_lines,
                        line_ts,
                        {i: line_assignments.get(i, []) for i in range(len(lyrics_lines))},
                        segments[0]['audio_path'] if segments else None,
                        Path(out_jsonl),
                        Path(out_log)
                    )
                else:
                    # Fallback: no lyrics file; use raw tmp_jsonl as aligned_jsonl (we'll let lrc_writer handle structure)
                    out_jsonl = tmp_jsonl
                    # Count entries
                    try:
                        with open(tmp_jsonl, "r", encoding="utf-8") as fh:
                            entries_list = [1 for _ in fh if _.strip()]
                        entries = len(entries_list)
                    except Exception:
                        entries = 0

                model_evals.append({
                    "model": model_name,
                    "raw_jsonl": tmp_jsonl,
                    "aligned_jsonl": out_jsonl,
                    "entries_count": entries
                })
            except Exception as e:
                mutils.dbg(f"Error evaluating model {model_name}: {e}")
                model_evals.append({
                    "model": model_name,
                    "raw_jsonl": tmp_jsonl if 'tmp_jsonl' in locals() else None,
                    "aligned_jsonl": None,
                    "entries_count": 0
                })

        # Compute the detailed scores using decision_logic
        enriched = decision_logic.compute_scores(model_evals, lyrics_txt_path if lyrics_txt_path else None)

        # Build scores string for logs with detailed components
        ordered_models = TRANSCRIBE_MODELS[:]
        for e in enriched:
            if e.get("model") not in ordered_models:
                ordered_models.append(e.get("model"))
        scores_parts = []
        for mn in ordered_models:
            found = next((m for m in enriched if m.get("model") == mn), None)
            if found:
                # format: model=score(coinc=jsonl_only=txt_only=mult, repeats=...)
                scores_parts.append(
                    f"{mn}={found.get('score',0.0):.3f}"
                    f"(coinc={found.get('coincidentes',0)},jsonl_only={found.get('jsonl_only',0)},txt_only={found.get('txt_only',0)},mult={found.get('mult',1.0):.1f},repeats={found.get('repeat_occurrences',0)})"
                )
        scores_str = ",".join(scores_parts)

        # Choose best model based on the computed scores
        # If we are in fallback mode (no lyrics file) pass flag to choose by repeats
        fallback_mode = (lyrics_txt_path is None and FALLBACK_TO_TRANSCRIPTION_ON_MISSING_LYRICS)
        best = decision_logic.choose_best_model(enriched, fallback_to_transcription=fallback_mode)
        if not best:
            append_anchor_entry(run_folder, basename, "no_best_model", f"scores={scores_str}")
            continue

        chosen_model = best.get("model")
        chosen_score = best.get("_decision_score", 0.0)

        # Log chosen model and scores (note if fallback used)
        mode_note = "fallback_transcription" if fallback_mode else "normal"
        append_anchor_entry(run_folder, basename, f"chosen:{chosen_model}", f"mode={mode_note}\tscores={scores_str}\tchosen_score={chosen_score:.3f}")

        # Write final LRC with chosen aligned jsonl (name LRC as <flac_base>.lrc)
        try:
            best_jsonl = best.get("aligned_jsonl")
            if best_jsonl and os.path.exists(best_jsonl):
                lrc_out = os.path.join(LYRICS_FOLDER, f"{flac_base}.lrc")
                entries = lrc_writer.read_jsonl(best_jsonl)
                lrc_writer.write_lrc(entries, lrc_out)

                # postprocess: split capitalized intralines and interpolate timestamps
                try:
                    from modules.lrc_postprocess import split_capitalized_intralines, append_trailing_empty_line
                    split_capitalized_intralines(lrc_out, out_path=None, fallback_dur=5.0)
                    mutils.dbg(f"Post-processed LRC (capital split) for: {lrc_out}")

                    # append a trailing empty timestamp line (last + 20s)
                    append_trailing_empty_line(lrc_out, extra_seconds=20.0, out_path=None)
                    mutils.dbg(f"Appended trailing empty timestamp to: {lrc_out}")
                except Exception as e:
                    mutils.dbg(f"Postprocess failed for {lrc_out}: {e}")

                # Copy definitive LRC to DB using canonical metadata name
                if artist and title:
                    db_manager.copy_to_lyrics_db(lrc_out, title, artist)
            else:
                append_anchor_entry(run_folder, basename, "no_aligned_output", f"scores={scores_str}")
        except Exception as e:
            append_anchor_entry(run_folder, basename, "write_lrc_failed", str(e))

    print("Done processing all files.")

if __name__ == "__main__":
    process_all()
