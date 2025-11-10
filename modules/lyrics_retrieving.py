"""
modules/lyrics_retrieving.py
Retrieve lyrics from Genius with candidate scanning, remix handling and artist strictness.

Responsibilities:
 - use search_song(title, artist) first.
 - if that fails, use search_songs(query) and rank candidates.
 - skip remix candidates unless core title matches closely.
 - require at least one strong artist match among candidates.
 - save retrieved raw lyrics to provided run_folder as "<Title> - <Artist>__genius.txt".
 - never write directly into lyrics/ from this module (run_folder only).
"""

import os
import re
from typing import Optional, List, Dict
from modules import utils as mutils
from config import CREDENTIALS_PATH
from lyricsgenius import Genius

SEARCH_CANDIDATES = 10

RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", flags=re.UNICODE)
REMIX_TOKENS = [
    "remix", "edit", "mix", "version", "rework", "bootleg", "radio edit",
    "vip", "extended", "feat", "featuring", "live", "acoustic", "instrumental"
]

def _normalize_for_compare(s: Optional[str]) -> str:
    """
    Normalize a string for robust title/artist comparisons.
    """
    if not s:
        return ""
    s2 = s.lower().strip()
    s2 = re.sub(r"[^\w\s']+", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    s2 = re.sub(r"^the\s+", "", s2)
    return s2

def _remove_parentheticals(s: str) -> str:
    """
    Remove text inside parentheses/brackets/braces to simplify titles.
    """
    return re.sub(r"[\(\[\{].*?[\)\]\}]", " ", s).strip()

def _contains_remix_token(s: str) -> bool:
    """
    Detect if a title contains remix/version tokens.
    """
    low = s.lower()
    for t in REMIX_TOKENS:
        if re.search(r'\b' + re.escape(t) + r'\b', low):
            return True
    return False

def _strip_remix_tokens(s: str) -> str:
    """
    Remove remix/version tokens and common separators from a title string.
    """
    s1 = re.sub(r'[\(\[\{].*?[\)\]\}]', ' ', s)
    pattern = r'\b(' + '|'.join(re.escape(t) for t in REMIX_TOKENS) + r')\b'
    s1 = re.sub(pattern, ' ', s1, flags=re.I)
    s1 = re.sub(r'[-_/]+', ' ', s1)
    s1 = re.sub(r'\s+', ' ', s1).strip()
    return s1

def _compute_similarity(a: str, b: str) -> float:
    """
    Compute similarity ratio between two strings using SequenceMatcher.
    """
    from difflib import SequenceMatcher
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def get_genius_client(token: str) -> Optional['Genius']:
    """
    Initialize and return a Genius client using the provided token.
    """
    if Genius is None:
        mutils.dbg("lyrics_retrieving: lyricsgenius not installed")
        return None
    try:
        g = Genius(token, timeout=15, retries=3, remove_section_headers=True)
        g.skip_non_songs = True
        g.verbose = False
        return g
    except Exception as e:
        mutils.dbg(f"lyrics_retrieving: Genius init error: {e}")
        return None

def _artist_tokens(artist: Optional[str]) -> List[str]:
    """
    Split an artist string into normalized tokens for matching.
    """
    if not artist:
        return []
    a = str(artist)
    a = re.sub(r'(?i)\s*(?:feat\.?|ft\.?|featuring)\b.*$', '', a)
    parts = re.split(r'\s*(?:,|&|\/| x | and )\s*', a)
    return [p.strip().lower() for p in parts if p and p.strip()]

def _artist_matches(requested_artist: str, returned_artist: Optional[str]) -> bool:
    """
    Return True when returned_artist matches requested_artist sufficiently.
    """
    if not requested_artist or not returned_artist:
        return False
    req_tokens = _artist_tokens(requested_artist)
    ret = (returned_artist or "").lower()
    for t in req_tokens:
        if t in ret:
            return True
        if _compute_similarity(t, ret) >= 0.85:
            return True
    if _compute_similarity(requested_artist.lower(), ret) >= 0.85:
        return True
    return False

def _rank_candidates(hits: List[Dict], req_title: str, req_artist: str) -> List[Dict]:
    """
    Rank search hits by title/artist similarity and remix preference.
    """
    ranked = []
    norm_req_title = _normalize_for_compare(_remove_parentheticals(req_title))
    for hit in hits:
        res = hit.get("result") if isinstance(hit, dict) else hit
        cand_title = (res.get("title") or "")
        cand_artist = ((res.get("primary_artist") or {}).get("name") or "")
        cand_title_no_paren = _remove_parentheticals(cand_title)
        cand_core = _strip_remix_tokens(cand_title_no_paren)
        is_remix = _contains_remix_token(cand_title)
        title_sim = _compute_similarity(_normalize_for_compare(cand_core), norm_req_title)
        artist_sim = _compute_similarity(_normalize_for_compare(cand_artist), _normalize_for_compare(req_artist))
        combined = 0.8 * title_sim + 0.2 * artist_sim
        ranked.append({
            "id": res.get("id"),
            "cand_title": cand_title,
            "cand_artist": cand_artist,
            "title_sim": title_sim,
            "artist_sim": artist_sim,
            "combined": combined,
            "is_remix": is_remix,
            "core_title_norm": _normalize_for_compare(cand_core)
        })
    # prefer non-remix then high combined score
    ranked.sort(key=lambda x: ((1 if x["is_remix"] else 0), -x["combined"], -x["title_sim"], -x["artist_sim"]))
    return ranked

def fetch_lyrics_by_meta(artist: str, title: str, genius: 'Genius', run_folder: Optional[str]) -> str:
    """
    Try multiple Genius API strategies to fetch lyrics for artist/title.
    """
    if genius is None:
        raise RuntimeError("Genius client not available")

    mutils.info(f"  -> Trying search_song(title='{title}', artist='{artist}')")
    try:
        song = genius.search_song(title, artist)
        if song and getattr(song, "lyrics", None):
            returned_artist = getattr(getattr(song, "primary_artist", None), "name", None)
            if _artist_matches(artist, returned_artist):
                mutils.info("  -> Lyrics obtained via direct search_song (artist match)")
                return song.lyrics
            mutils.dbg("  -> direct search_song artist mismatch; will scan candidates")
    except Exception as e:
        mutils.dbg(f"  -> search_song direct raised: {e}")

    # Now search candidates
    q = f"{title} {artist}"
    mutils.dbg(f"  -> search_songs(query='{q}') asking for {SEARCH_CANDIDATES} hits")
    try:
        results = genius.search_songs(q, per_page=SEARCH_CANDIDATES)
        hits = results.get("hits", []) if isinstance(results, dict) else []
        if not hits:
            raise RuntimeError("No search hits")

        ranked = _rank_candidates(hits, title, artist)

        # Strict rule: require exact (normalized) core title match among candidates first.
        norm_req_core = _normalize_for_compare(_strip_remix_tokens(_remove_parentheticals(title)))
        exact_title_candidates = [r for r in ranked if r.get("core_title_norm") == norm_req_core]
        if not exact_title_candidates:
            raise RuntimeError("No candidate with exact title match found")

        # From exact-title matches, require at least one with a matching artist token.
        candidates_with_artist = [r for r in exact_title_candidates if _artist_matches(artist, r.get("cand_artist"))]
        if not candidates_with_artist:
            raise RuntimeError("No candidate with exact title and matching artist found")

        tried = []
        last_errs = []
        for cand in candidates_with_artist:
            tried.append(f"{cand.get('cand_title')} - {cand.get('cand_artist')} (remix={cand.get('is_remix')})")
            cand_title = cand.get('cand_title')
            cand_artist = cand.get('cand_artist')
            cand_id = cand.get('id')

            # Skip remix if core title mismatch to save attempts
            if cand.get('is_remix'):
                norm_req_core = _normalize_for_compare(_strip_remix_tokens(_remove_parentheticals(title)))
                if cand.get('core_title_norm') != norm_req_core and cand.get('title_sim', 0.0) < 0.90:
                    mutils.dbg(f"  -> skipping remix candidate (core mismatch): {cand_title} - {cand_artist}")
                    continue

            # Attempt 1: search_song(cand_title, cand_artist)
            try:
                mutils.dbg(f"  -> Trying search_song for candidate: '{cand_title}' - '{cand_artist}'")
                cand_song = genius.search_song(cand_title, cand_artist)
                if cand_song and getattr(cand_song, "lyrics", None):
                    returned_artist = getattr(getattr(cand_song, "primary_artist", None), "name", None)
                    if _artist_matches(artist, returned_artist):
                        mutils.info("  -> Lyrics obtained using search_song on candidate")
                        return cand_song.lyrics
                    mutils.dbg("    candidate returned_artist does not match enough -> continue attempts")
            except Exception as e:
                mutils.dbg(f"    search_song(candidate) raised: {e}")
                last_errs.append(str(e))

            # Attempt 2: genius.song(id) -> then try path/url via genius.lyrics
            try:
                if cand_id:
                    mutils.dbg(f"  -> Trying genius.song(id={cand_id}) to retrieve path/url")
                    song_data = genius.song(cand_id)
                    if isinstance(song_data, dict) and 'song' in song_data:
                        sd = song_data['song']
                        lyrics = sd.get('lyrics') or sd.get('lyrics_body') or ""
                        if lyrics and len(lyrics.strip()) > 30:
                            returned_artist = sd.get('primary_artist', {}).get('name')
                            if _artist_matches(artist, returned_artist):
                                mutils.info("  -> Lyrics obtained via song(id) field")
                                return lyrics
                        path = sd.get('path') or sd.get('url') or ""
                        if path:
                            if not path.startswith("http"):
                                if not path.startswith("/"):
                                    path = "/" + path
                                full_url = "https://genius.com" + path
                            else:
                                full_url = path
                            mutils.dbg(f"    will try genius.lyrics(url='{full_url}')")
                            try:
                                page_lyrics = genius.lyrics(song_url=full_url)
                                if page_lyrics and len(page_lyrics.strip()) > 20:
                                    mutils.info("  -> Lyrics obtained by scraping candidate URL (genius.lyrics)")
                                    return page_lyrics
                            except Exception as e:
                                mutils.dbg(f"    genius.lyrics(url) failed: {e}")
                                last_errs.append(str(e))
            except Exception as e:
                mutils.dbg(f"  -> genius.song(song_id) step raised: {e}")
                last_errs.append(str(e))

        # exhausted candidates_with_artist
        raise RuntimeError(f"No valid lyrics after scanning {len(tried)} artist-matching candidates. Tried examples: {tried[:6]} Errors: {last_errs[:3]}")
    except Exception as e:
        mutils.dbg(f"  -> search_songs error/insufficient: {e}")
        raise RuntimeError(f"All Genius methods failed: {e}")

def retrieve_for_flac(flac_path: str, run_folder: Optional[str] = None) -> Optional[str]:
    """
    Retrieve and save raw Genius lyrics for a FLAC file based on metadata.
    """
    artist, title = mutils.extract_metadata_from_flac(flac_path)
    mutils.info(f"Processing: {os.path.basename(flac_path)} metadata -> artist: '{artist}' | title: '{title}'")
    if not artist or not title:
        mutils.dbg("lyrics_retrieving: metadata incomplete; skipping (do not use filename).")
        return None

    creds = mutils.load_credentials(CREDENTIALS_PATH)
    token = None
    if isinstance(creds, dict):
        for key in ['genius_access_token', 'genius_token', 'genius']:
            if key in creds and creds[key]:
                token = creds[key]
                break
    if not token:
        mutils.info("No Genius token available; skipping Genius retrieval.")
        return None

    genius = get_genius_client(token)
    if not genius:
        mutils.info("Genius client unavailable; skipping lyrics retrieval.")
        return None

    try:
        lyrics = fetch_lyrics_by_meta(artist, title, genius, run_folder)
        if lyrics:
            try:
                os.makedirs(run_folder, exist_ok=True)
                outname = mutils.sanitize_filename(f"{title} - {artist}__genius.txt")
                outpath = os.path.join(run_folder, outname)
                with open(outpath, "w", encoding="utf-8") as fo:
                    fo.write(lyrics)
                mutils.dbg(f"Saved Genius raw lyrics to run logs: {outpath}")
                return outpath
            except Exception as e:
                mutils.dbg(f"Could not save lyrics to run_folder: {e}")
                return None
    except Exception as e:
        mutils.dbg(f"Could not retrieve lyrics for '{os.path.basename(flac_path)}': {e}")
        return None

    return None
