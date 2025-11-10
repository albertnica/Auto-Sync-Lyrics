# Toggle debug verbosity. When DEBUG is True many extra details are printed to stdout.
DEBUG = True

# Directory names
SONGS_FOLDER = "songs"           # Where your audio files (.flac) live
LYRICS_FOLDER = "lyrics"         # Output folder where generated .lrc files are written
LYRICS_DB_FOLDER = ".lyrics_db"  # Persistent database of processed lyrics (normalized: "<Title> - <Artist>.lrc")
LOGS_FOLDER = ".logs"            # Root logs folder

# Transcription models to try (order = preference/try order)
TRANSCRIBE_MODELS = ["openai/whisper-large-v3-turbo", "openai/whisper-medium", "openai/whisper-small"]

# Paths / filenames
CREDENTIALS_PATH = "credentials.json"    # Optional (Genius token etc.)
LOG_NAME = ".log.txt"

# If True, when lyrics are not found on Genius
# the pipeline continues using the transcriptions (it chooses
# the transcription with fewer repeated words). If False (default),
# the run is aborted when lyrics are missing.
FALLBACK_TO_TRANSCRIPTION_ON_MISSING_LYRICS = False
