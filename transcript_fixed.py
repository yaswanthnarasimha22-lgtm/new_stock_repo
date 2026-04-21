#!/usr/bin/env python3
"""
Travel Call TRANSCRIPTION — Batch API Version (Vertex AI Gemini Batch Inference)
==================================================================================
BATCH PROCESSING VERSION — Direct date folder processing, no scheduler

Pipeline:
  Phase 1 (local GPU/CPU): Download → Preprocess (inaSpeechSegmenter + VAD) → Chunk
  Phase 2 (cloud):         Upload chunks to GCS → Build JSONL → Submit batch job →
                           Poll → Parse results → Stitch → Save

Key Features:
  - Gemini Batch API: 50% cost savings, no rate limiting / 429 errors
  - Multi-date support: --date 2026-02-28 2026-03-01 processes both sequentially
  - Direct date folder: recordings sit directly in teleappliant_recordings/YYYY-MM-DD/
  - DID & Extension filtering from dids.txt / extensions.txt
  - Processed file tracking via processed_calls.json (no reprocessing)
  - GPU preprocessing (inaSpeechSegmenter + VAD) — works on 8GB VRAM (~1-2GB used)
  - Agent name correction from agent_names.txt
  - Transcript duration verification
  - Hallucination detection & chunk validation
  - Timestamp remapping to original audio timeline
  - DAEMON MODE: --daemon flag

Hardware (8GB VRAM target):
  - inaSpeechSegmenter CNN: ~1-2GB VRAM. 8GB is plenty.
  - webrtcvad / pydub / ffmpeg: CPU-only, fast.
  - 250 calls @ 8GB GPU: ~15-25 min preprocessing + ~30-120 min batch API = ~1-2.5 hrs total

Dependencies:
  pip install torch google-genai google-cloud-storage pydub webrtcvad inaSpeechSegmenter
  apt install ffmpeg
"""
import os
import sys
import json
import re
import time
import subprocess
import traceback
import hashlib
import argparse
import shutil
import threading
import signal
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch


# =======================================================================
# CONFIGURATION
# =======================================================================
class Config:
    """Configuration constants for the batch transcription pipeline."""
    # GCS Buckets & Paths
    LOCATION = "us-central1"
    INPUT_BUCKET = "travel-audio-batch-input"
    INPUT_BASE_FOLDER = "teleappliant_recordings"
    OUTPUT_BUCKET = "travel-audio-results-2409"
    OUTPUT_FOLDER = "results"
    LOCAL_RESULTS_DIR = "analysis_results"
    CREDENTIALS_PATH = "credentials.json"
    PROCESSED_TRACKER_FILE = "processed_calls.json"
    LOG_DIR = "logs"

    # Batch API GCS paths
    BATCH_INPUT_FOLDER = "batch_inputs"
    BATCH_OUTPUT_FOLDER = "batch_outputs"
    BATCH_CHUNKS_FOLDER = "batch_chunks"

    # Filter Files
    DIDS_FILE = "dids.txt"
    EXTENSIONS_FILE = "extensions.txt"
    AGENT_NAMES_FILE = "agent_names.txt"

    # Preprocessing (tuned values — do not change)
    FRAME_DURATION_MS = 30
    VAD_AGGRESSIVENESS = 1
    HIGH_PASS_HZ = 100
    LOW_PASS_HZ = 4000
    MIN_SPEECH_DURATION_MS = 500

    # Chunking
    CHUNK_DURATION_MINUTES = 15
    CHUNK_OVERLAP_SECONDS = 30
    SHORT_AUDIO_THRESHOLD_MINUTES = 8
    MAX_CHUNKS_PER_FILE = 30

    # Hallucination Detection
    MAX_LINE_LENGTH = 500
    REPETITION_WINDOW = 5
    REPETITION_SIMILARITY = 0.75
    MIN_CHUNK_WORDS = 20
    MIN_UNIQUE_WORD_RATIO = 0.5

    # Batch API Settings
    GEMINI_MODEL = "gemini-2.5-flash"
    TRANSCRIBE_MAX_TOKENS = 65536
    TEMPERATURE = 0.1
    BATCH_POLL_INTERVAL = 60            # Poll every 60s (was 30 — less noise in logs)
    BATCH_POLL_TIMEOUT = 86400           # 24 hours (was 2h — batch jobs need 2-4h for 250+ audio chunks)
    BATCH_MAX_RETRIES = 2

    # Parallelism — preprocessing only (batch API handles transcription)
    DEFAULT_WORKERS = 2
    MAX_WORKERS = 5

    # Audio Formats
    AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".aac", ".wma", ".webm", ".opus")
    AUDIO_MIME_TYPES = {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".mp4": "video/mp4", ".aac": "audio/aac", ".wma": "audio/x-ms-wma",
        ".webm": "audio/webm", ".opus": "audio/opus",
    }


# =======================================================================
# GLOBAL STATE
# =======================================================================
DEVICE_INFO = {}
DEVICE = "cpu"
PROJECT_ID = None
shutdown_requested = False

_save_lock = threading.Lock()
_log_lock = threading.Lock()
_logger = None

_seg = None
_seg_device = None
_seg_lock = threading.Lock()


# =======================================================================
# LOGGING
# =======================================================================
def setup_logging(daemon_mode: bool = False) -> logging.Logger:
    """Configure dual logging (terminal + file)."""
    global _logger
    if _logger is not None:
        return _logger

    log_dir = Path(Config.LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"transcription_{timestamp}.log"

    logger = logging.getLogger("travel_transcription")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(threadName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if not daemon_mode:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if daemon_mode:
        try:
            import systemd.journal
            jh = systemd.journal.JournalHandler()
            jh.setLevel(logging.INFO)
            jh.setFormatter(logging.Formatter('%(levelname)s %(threadName)s: %(message)s'))
            logger.addHandler(jh)
        except ImportError:
            pass

    _logger = logger
    return logger


def log(msg: str, level: str = "INFO"):
    """Thread-safe logging with emoji prefixes."""
    logger = setup_logging()

    level_map = {
        "INFO": logging.INFO, "OK": logging.INFO, "WARN": logging.WARNING,
        "ERROR": logging.ERROR, "PROCESSING": logging.INFO, "UPLOAD": logging.INFO,
        "DOWNLOAD": logging.INFO, "AUDIO": logging.INFO, "SAVE": logging.INFO,
        "WAIT": logging.INFO, "DONE": logging.INFO, "BATCH": logging.INFO,
        "MEGA": logging.INFO, "PREPROCESS": logging.INFO, "VAD": logging.INFO,
        "MAP": logging.INFO, "GPU": logging.INFO, "API": logging.INFO,
        "TRACK": logging.INFO, "HOUR": logging.INFO,
    }

    emoji_map = {
        "INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "PROCESSING": "🔄",
        "UPLOAD": "📤", "DOWNLOAD": "📥", "AUDIO": "🎤", "SAVE": "💾", "WAIT": "⏳",
        "DONE": "🎉", "BATCH": "📦", "MEGA": "🚀", "PREPROCESS": "🎵", "VAD": "🔇",
        "MAP": "🗺️", "GPU": "🎮", "API": "🔶", "TRACK": "📋", "HOUR": "⏰",
    }

    log_level = level_map.get(level, logging.INFO)
    emoji = emoji_map.get(level, "•")
    logger.log(log_level, msg)

    if not getattr(setup_logging, "daemon_mode", False):
        with _log_lock:
            sys.stdout.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {emoji} {msg}\n")
            sys.stdout.flush()


setup_logging.daemon_mode = False


# =======================================================================
# DAEMON SUPPORT
# =======================================================================
def become_daemon():
    """Fork process to background."""
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
        os.setsid()
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
        os.chdir("/")
        os.umask(0)
        sys.stdin.close()
        sys.stdout.close()
        sys.stderr.close()
        sys.stdin = open(os.devnull, 'r')
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        setup_logging.daemon_mode = True
        setup_logging(daemon_mode=True)
        log("Process daemonized successfully", "INFO")
        return True
    except Exception as e:
        print(f"Daemonization failed: {e}", file=sys.stderr)
        return False


# =======================================================================
# DEVICE & SETUP
# =======================================================================
def get_device_info():
    """Detect GPU availability."""
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "vram_free_gb": round((torch.cuda.get_device_properties(0).total_memory -
                                 torch.cuda.memory_allocated(0)) / (1024**3), 2),
            "cuda_version": torch.version.cuda,
        }
    return {"device": "cpu", "gpu_name": None, "vram_total_gb": 0, "vram_free_gb": 0, "cuda_version": None}


def _load_project_id():
    """Load GCP project ID from credentials."""
    global PROJECT_ID
    if not os.path.exists(Config.CREDENTIALS_PATH):
        log(f"ERROR: {Config.CREDENTIALS_PATH} not found", "ERROR")
        sys.exit(1)
    with open(Config.CREDENTIALS_PATH) as f:
        creds = json.load(f)
    PROJECT_ID = creds.get("project_id")
    if not PROJECT_ID:
        log("ERROR: No project_id in credentials.json", "ERROR")
        sys.exit(1)
    log(f"[CONFIG] Project: {PROJECT_ID} | SA: {creds.get('client_email', '?')}", "INFO")
    log(f"[CONFIG] Buckets: in={Config.INPUT_BUCKET}, out={Config.OUTPUT_BUCKET}", "INFO")


DEVICE_INFO = get_device_info()
DEVICE = DEVICE_INFO["device"]
_load_project_id()


# =======================================================================
# UTILITY FUNCTIONS
# =======================================================================
def format_file_size(bytes_size: Optional[int]) -> str:
    if bytes_size is None:
        return "?"
    if bytes_size < 1024:
        return f"{bytes_size}B"
    if bytes_size < 1048576:
        return f"{bytes_size/1024:.1f}KB"
    return f"{bytes_size/1048576:.1f}MB"


def format_duration(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m {s%60}s"
    return f"{s//60}m {s%60}s"


def get_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def print_gpu_status():
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        log(f"VRAM: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved / {DEVICE_INFO['vram_total_gb']}GB", "GPU")


def clear_gpu_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =======================================================================
# FILTERING & AGENT NAMES
# =======================================================================
def load_filter_list(filepath: str, label: str) -> List[str]:
    """Load filter list from file (one item per line)."""
    if not os.path.exists(filepath):
        log(f"{label} file not found: {filepath} — no {label} filter active", "WARN")
        return []
    items = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(line)
    if items:
        log(f"Loaded {len(items)} {label} from {filepath}", "OK")
        for item in items[:5]:
            log(f"  • {item}", "INFO")
        if len(items) > 5:
            log(f"  ... and {len(items)-5} more", "INFO")
    return items


def filename_matches_filters(filename: str, dids: List[str], extensions: List[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if filename matches any DID or extension."""
    for did in dids:
        if did in filename:
            return True, "DID", did
    for ext in extensions:
        if ext in filename:
            return True, "EXT", ext
    return False, None, None


def load_agent_names(filepath: str = Config.AGENT_NAMES_FILE) -> List[str]:
    """Load agent names for spelling correction."""
    if not os.path.exists(filepath):
        log(f"Agent names file not found: {filepath}", "WARN")
        return []
    names = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                name = line.split(',')[0].strip() if ',' in line else line
                names.append(name)
    if names:
        log(f"Loaded {len(names)} agent names from {filepath}", "OK")
        for name in names[:5]:
            log(f"  • {name}", "INFO")
        if len(names) > 5:
            log(f"  ... and {len(names)-5} more", "INFO")
    return names


# =======================================================================
# TRANSCRIPTION PROMPT (UPDATED agent name rules)
# =======================================================================
def get_transcription_prompt(agent_names: Optional[List[str]] = None) -> str:
    """Generate transcription prompt with agent name correction."""
    prompt = """You are a precise audio transcriptionist for a travel call center. The brand is Teletext Holidays, and the website is teletextholidays.co.uk.

TASK: Transcribe this audio segment COMPLETELY — every single word, exactly as spoken.

CRITICAL RULES:
1. Provide the COMPLETE word-for-word transcript. Do NOT skip, summarize, or paraphrase ANY part.
2. Label speakers as "Agent:" and "Customer:" (or "Agent 1:", "Agent 2:" if multiple agents).
3. Include timestamps relative to THIS audio segment starting from [00:00].
   - Use format [MM:SS] for times under 1 hour, [H:MM:SS] for times over 1 hour.
   - Add a timestamp at least every 30 seconds of speech, and at every speaker change.
   - Each new timestamp MUST be on its own new line.
4. Redact credit card numbers, CVV codes, expiration dates, and full street addresses as [PII REDACTED].
5. Do NOT redact phone numbers, email addresses, or names — keep them exactly as spoken.
6. Capture filler words (um, uh, like), false starts, and interruptions accurately.
7. If speakers talk over each other, note it as [crosstalk].
8. If there is silence or hold music, note it briefly (e.g., [hold music ~2min], [silence ~30s]).
9. Do NOT add any analysis, summary, or commentary — ONLY the transcript.
10. Do NOT skip any part of the audio even if it seems repetitive or unimportant.
11. Transcribe the ENTIRE audio from start to finish with NO gaps.
12. IMPORTANT: Put EACH speaker turn on its own line. Do NOT put the entire transcript on a single line.
13. NEVER repeat the same phrase or sentence. If you notice yourself repeating, STOP and move forward.
14. If the transcript appears incomplete due to audio cutoff, continue transcribing until the audio segment ends completely."""

    if agent_names and len(agent_names) > 0:
        agent_names_str = "\n".join(f"- {name}" for name in agent_names)
        prompt += f"""

AGENT NAME SPELLING REFERENCE:
{agent_names_str}

AGENT NAME RULES — READ CAREFULLY:
When transcribing, ensure that agent names are spelled exactly as provided in the predefined list, paying close attention to unusual spellings or pronunciations.
If an agent introduces themselves, use the correct spelling from the list. If the spoken name is a very close match to a name in the list (same starting letter, similar number of syllables, and only minor phonetic differences), normalize it to the closest matching name from the list—for example, 
if you hear "Sumeer" and the list contains "Sumeer," output "Sumeer"; if you hear "Hayley" and the list contains "Hailey," output "Hailey"; if you hear "Zak" and the list contains "Zack," output "Zack." Do not invent new names and always prefer the closest valid match from the provided list. 
If the agent does not introduce themselves at all, label them simply as "Agent:" without a name. This list should only be used to correct minor spelling variations of names that are actually spoken and must not be treated as a list to choose names from.
"""

    prompt += """

OUTPUT FORMAT — one speaker turn per line, nothing else:
[00:00] Agent: ...
[00:05] Customer: ...
[00:30] Agent: ...
"""
    return prompt


# =======================================================================
# CHUNK VALIDATION & HALLUCINATION DETECTION
# =======================================================================
def is_valid_chunk(transcript: str) -> bool:
    """Validate chunk transcript quality."""
    words = transcript.split()
    if len(words) < Config.MIN_CHUNK_WORDS:
        return False
    unique_ratio = len(set(words)) / (len(words) + 1)
    if unique_ratio < Config.MIN_UNIQUE_WORD_RATIO:
        return False
    if transcript.endswith("...") or transcript.endswith("…"):
        return False
    return True


def clean_chunk_transcript(raw_transcript: str) -> str:
    """Remove hallucinations and repetitions safely."""
    if not raw_transcript:
        return raw_transcript

    lines = []
    for raw_line in raw_transcript.split('\n'):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        ts_positions = [m.start() for m in re.finditer(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', raw_line)]
        if len(ts_positions) > 1:
            for i, pos in enumerate(ts_positions):
                end = ts_positions[i+1] if i+1 < len(ts_positions) else len(raw_line)
                segment = raw_line[pos:end].strip()
                if segment:
                    lines.append(segment)
        else:
            lines.append(raw_line)

    capped_lines = []
    for line in lines:
        if len(line) > Config.MAX_LINE_LENGTH:
            truncated = line[:Config.MAX_LINE_LENGTH]
            last_space = truncated.rfind(' ')
            if last_space > Config.MAX_LINE_LENGTH * 0.7:
                truncated = truncated[:last_space]
            capped_lines.append(truncated)
            log(f"    ⚠️ Truncated hallucination line: {len(line)} → {len(truncated)} chars", "WARN")
        else:
            capped_lines.append(line)

    cleaned = []
    repetition_count = 0
    repetition_truncated = False

    for i, line in enumerate(capped_lines):
        if repetition_truncated:
            break
        current_norm = _normalize_text(line)
        is_repeat = False
        if current_norm and len(current_norm) > 10 and len(cleaned) >= 2:
            recent_norms = [_normalize_text(l) for l in cleaned[-Config.REPETITION_WINDOW:]]
            for recent in recent_norms:
                if recent and len(recent) > 10 and _text_similarity(current_norm, recent) > Config.REPETITION_SIMILARITY:
                    is_repeat = True
                    break
        if is_repeat:
            repetition_count += 1
            if repetition_count >= Config.REPETITION_WINDOW:
                log(f"    ⚠️ Repetition detected at line {i}, skipping", "WARN")
                repetition_truncated = True
                continue
        else:
            repetition_count = 0
        cleaned.append(line)

    result = '\n'.join(cleaned)
    if len(result) < len(raw_transcript) * 0.5 and len(raw_transcript) > 1000:
        log(f"    ℹ️ Cleaned: {len(raw_transcript)} → {len(result)} chars", "INFO")
    return result


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'(agent|customer)\s*\d*\s*:', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# =======================================================================
# AUDIO PREPROCESSING (GPU/CPU)
# =======================================================================
def get_segmenter():
    """Get thread-safe inaSpeechSegmenter instance."""
    global _seg, _seg_device
    with _seg_lock:
        if _seg is None:
            target = DEVICE
            log(f"Loading inaSpeechSegmenter ({target.upper()})...",
                "GPU" if target == "cuda" else "PREPROCESS")
            if target == "cuda":
                vram_before = torch.cuda.memory_allocated(0) / (1024**3)
                log(f"  VRAM before: {vram_before:.2f}GB", "GPU")
            try:
                _seg = Segmenter()
                _seg_device = target
                if target == "cuda":
                    vram_after = torch.cuda.memory_allocated(0) / (1024**3)
                    delta = vram_after - vram_before
                    if delta > 0.1:
                        log(f"  Loaded on GPU (VRAM: +{delta:.2f}GB)", "GPU")
                    else:
                        log(f"  Loaded (VRAM +{delta:.2f}GB — may be on CPU)", "WARN")
                        _seg_device = "cpu"
                else:
                    log("  Loaded on CPU", "OK")
            except Exception as e:
                log(f"  Load failed: {e}", "WARN")
                if target == "cuda":
                    log("  Falling back to CPU...", "INFO")
                    _seg = Segmenter()
                    _seg_device = "cpu"
                    log("  Loaded on CPU (fallback)", "OK")
                else:
                    raise
    return _seg


def run_segmenter(path: str):
    with _seg_lock:
        return _seg(path)


def filter_music(path: str) -> Tuple[Any, List[Tuple[int, int]]]:
    """Remove music segments using inaSpeechSegmenter."""
    log("    Removing music..." + (" (GPU)" if _seg_device == "cuda" else ""),
        "GPU" if _seg_device == "cuda" else "PREPROCESS")
    start_time = time.time()
    segments = run_segmenter(path)
    elapsed = time.time() - start_time

    orig = AudioSegment.from_file(path)
    speech = AudioSegment.empty()
    kept_segments = []

    for label, start, end in segments:
        if label in ['male', 'female']:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            speech += orig[start_ms:end_ms]
            kept_segments.append((start_ms, end_ms))

    log(f"    {len(orig)}ms→{len(speech)}ms ({len(kept_segments)} segs) [{elapsed:.1f}s]", "PREPROCESS")
    return speech, kept_segments


def vad_filter(audio: Any, aggressiveness: int = Config.VAD_AGGRESSIVENESS) -> Tuple[Any, List[Tuple[int, int]]]:
    """Apply VAD filtering."""
    vad = webrtcvad.Vad(aggressiveness)
    pcm_data, sample_rate = _audio_to_pcm(audio)
    frame_duration = Config.FRAME_DURATION_MS
    frame_size = int(sample_rate * frame_duration / 1000) * 2
    frames = [pcm_data[i:i+frame_size] for i in range(0, len(pcm_data), frame_size)]

    filtered_audio = AudioSegment.empty()
    kept_ranges = []
    for i, frame in enumerate(frames):
        if len(frame) < frame_size:
            continue
        if vad.is_speech(frame, sample_rate):
            start_ms = i * frame_duration
            end_ms = start_ms + frame_duration
            filtered_audio += audio[start_ms:end_ms]
            kept_ranges.append((start_ms, end_ms))
    return filtered_audio, kept_ranges


def _audio_to_pcm(audio: Any) -> Tuple[bytes, int]:
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    return audio.raw_data, 16000


def build_timestamp_map(kept_segments: List[Tuple[int, int]],
                       vad_kept: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Build mapping from processed time to original time."""
    music_map = []
    current_processed = 0
    for start_orig, end_orig in kept_segments:
        music_map.append((current_processed, start_orig))
        duration = end_orig - start_orig
        music_map.append((current_processed + duration, end_orig))
        current_processed += duration
    if not music_map:
        return []

    vad_map = []
    current_vad = 0
    for start_vad, end_vad in vad_kept:
        vad_map.append((current_vad, start_vad))
        duration = end_vad - start_vad
        vad_map.append((current_vad + duration, end_vad))
        current_vad += duration
    if not vad_map:
        return music_map

    final_map = []
    for vad_time, orig_time_after_music in vad_map:
        orig_time = _interpolate(music_map, orig_time_after_music)
        final_map.append((vad_time, orig_time))

    final_map = sorted(set(final_map), key=lambda x: x[0])
    if len(final_map) > 1000:
        step = max(1, len(final_map) // 1000)
        simplified = [final_map[i] for i in range(0, len(final_map), step)]
        if simplified[-1] != final_map[-1]:
            simplified.append(final_map[-1])
        final_map = simplified
    return final_map


def _interpolate(mapping: List[Tuple[int, int]], query_time: int) -> int:
    if not mapping:
        return query_time
    if query_time <= mapping[0][0]:
        return mapping[0][1]
    if query_time >= mapping[-1][0]:
        return mapping[-1][1]
    low, high = 0, len(mapping) - 1
    while low < high - 1:
        mid = (low + high) // 2
        if mapping[mid][0] <= query_time:
            low = mid
        else:
            high = mid
    proc1, orig1 = mapping[low]
    proc2, orig2 = mapping[high]
    if proc2 == proc1:
        return orig1
    ratio = (query_time - proc1) / (proc2 - proc1)
    return int(orig1 + ratio * (orig2 - orig1))


def preprocess_audio(path: str, filename: str, work_dir: str) -> Tuple[Optional[str], Optional[List], float, Optional[float]]:
    """Full preprocessing pipeline: music removal + VAD + filtering."""
    log(f"    Preprocessing: {filename}", "PREPROCESS")
    try:
        original = AudioSegment.from_file(path)
        orig_duration = len(original) / 1000.0
        log(f"    Original: {format_duration(orig_duration)}", "AUDIO")

        speech, kept_segments = filter_music(path)
        if len(speech) < Config.MIN_SPEECH_DURATION_MS:
            log("    Too short after music removal", "WARN")
            return None, None, orig_duration, None

        log(f"    Bandpass {Config.HIGH_PASS_HZ}-{Config.LOW_PASS_HZ}Hz...", "PREPROCESS")
        filtered = speech.high_pass_filter(Config.HIGH_PASS_HZ).low_pass_filter(Config.LOW_PASS_HZ)

        log(f"    VAD (agg={Config.VAD_AGGRESSIVENESS})...", "VAD")
        vad_audio, vad_kept = vad_filter(filtered, Config.VAD_AGGRESSIVENESS)
        if vad_audio is None or len(vad_audio) < Config.MIN_SPEECH_DURATION_MS:
            log("    Too short after VAD", "WARN")
            return None, None, orig_duration, None

        proc_duration = len(vad_audio) / 1000.0
        log(f"    Clean: {format_duration(proc_duration)} (removed {format_duration(orig_duration - proc_duration)})", "OK")
        timestamp_map = build_timestamp_map(kept_segments, vad_kept)
        log(f"    Map: {len(timestamp_map)} points", "MAP")

        prep_dir = os.path.join(work_dir, "preprocessed")
        os.makedirs(prep_dir, exist_ok=True)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(prep_dir, f"{base_name}_preprocessed.mp3")
        vad_audio.export(output_path, format="mp3", parameters=["-b:a", "64k", "-ar", "16000", "-ac", "1"])

        return output_path, timestamp_map, orig_duration, proc_duration
    except Exception as e:
        log(f"    Failed: {e}", "ERROR")
        traceback.print_exc()
        return None, None, 0.0, None


# =======================================================================
# CHUNKING
# =======================================================================
def get_audio_duration(filepath: str) -> Optional[float]:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def split_chunks(preprocessed_path: str, chunk_dir: str, filename: str) -> Tuple[List[Dict], Optional[float]]:
    """Split audio into overlapping chunks."""
    base_name = os.path.splitext(filename)[0]
    duration = get_audio_duration(preprocessed_path)

    if duration is None:
        return [{"path": preprocessed_path, "chunk_num": 1, "start_time": 0,
                "end_time": None, "original_filename": filename}], None

    if duration <= Config.SHORT_AUDIO_THRESHOLD_MINUTES * 60:
        return [{"path": preprocessed_path, "chunk_num": 1, "start_time": 0,
                "end_time": duration, "original_filename": filename}], duration

    chunk_duration_sec = Config.CHUNK_DURATION_MINUTES * 60
    overlap_sec = Config.CHUNK_OVERLAP_SECONDS
    step_sec = chunk_duration_sec - overlap_sec
    chunks = []
    chunk_num = 0
    start_time = 0

    chunk_base_dir = os.path.join(chunk_dir, base_name)
    os.makedirs(chunk_base_dir, exist_ok=True)

    while start_time < duration and chunk_num < Config.MAX_CHUNKS_PER_FILE:
        chunk_num += 1
        end_time = min(start_time + chunk_duration_sec, duration)
        actual_duration = end_time - start_time
        if actual_duration < 10 and chunk_num > 1:
            break

        chunk_filename = f"{base_name}_chunk{chunk_num:03d}.mp3"
        chunk_path = os.path.join(chunk_base_dir, chunk_filename)

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", preprocessed_path, "-ss", str(start_time),
                 "-t", str(actual_duration), "-acodec", "libmp3lame", "-b:a", "64k",
                 "-ar", "16000", "-ac", "1", "-loglevel", "error", chunk_path],
                check=True, capture_output=True, timeout=180
            )
            if os.path.exists(chunk_path):
                chunks.append({
                    "path": chunk_path, "chunk_num": chunk_num,
                    "start_time": start_time, "end_time": end_time,
                    "size": os.path.getsize(chunk_path),
                    "original_filename": filename,
                    "file_hash": get_file_hash(chunk_path)
                })
        except Exception as e:
            log(f"    Chunk {chunk_num} creation failed: {e}", "ERROR")
        start_time += step_sec

    return chunks, duration


# =======================================================================
# GCS INTEGRATION
# =======================================================================
_storage_client = None
_storage_client_lock = threading.Lock()


def get_storage_client():
    """Get cached GCS client (singleton — avoids re-auth on every call)."""
    global _storage_client
    with _storage_client_lock:
        if _storage_client is None:
            _storage_client = storage.Client.from_service_account_json(Config.CREDENTIALS_PATH)
        return _storage_client


def get_vertex_client():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(Config.CREDENTIALS_PATH)
    return genai.Client(
        http_options=HttpOptions(api_version="v1"),
        vertexai=True,
        project=PROJECT_ID,
        location=Config.LOCATION
    )


def download_from_gcs(blob_name: str, filename: str, download_dir: str) -> Optional[str]:
    os.makedirs(download_dir, exist_ok=True)
    local_path = os.path.join(download_dir, filename)
    if os.path.exists(local_path):
        return local_path
    try:
        client = get_storage_client()
        bucket = client.bucket(Config.INPUT_BUCKET)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        return local_path
    except Exception as e:
        log(f"    Download failed: {e}", "ERROR")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str) -> Optional[str]:
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        return f"gs://{bucket_name}/{gcs_path}"
    except Exception as e:
        log(f"    Upload to GCS failed: {e}", "ERROR")
        return None


def upload_string_to_gcs(content: str, bucket_name: str, gcs_path: str) -> Optional[str]:
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(content, content_type="application/jsonl")
        return f"gs://{bucket_name}/{gcs_path}"
    except Exception as e:
        log(f"    Upload string to GCS failed: {e}", "ERROR")
        return None


def download_string_from_gcs(bucket_name: str, gcs_path: str) -> Optional[str]:
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        return blob.download_as_text()
    except Exception as e:
        log(f"    Download string from GCS failed: {e}", "ERROR")
        return None


def list_gcs_blobs(bucket_name: str, prefix: str) -> List[str]:
    try:
        client = get_storage_client()
        blobs = client.list_blobs(bucket_name, prefix=prefix)
        return [b.name for b in blobs if not b.name.endswith("/")]
    except Exception as e:
        log(f"    List GCS blobs failed: {e}", "ERROR")
        return []


def list_audio_files_in_date_folder(date_folder: str, dids: List[str],
                                    extensions: List[str], tracker) -> List[Dict]:
    """
    List matching audio files directly in the date folder.
    Structure: gs://INPUT_BUCKET/teleappliant_recordings/YYYY-MM-DD/*.mp3
    No subfolders.
    """
    prefix = f"{Config.INPUT_BASE_FOLDER}/{date_folder}/"
    processed_blobs = tracker.get_processed_blobs()

    try:
        client = get_storage_client()
        all_blobs = list(client.list_blobs(Config.INPUT_BUCKET, prefix=prefix))

        audio_files = []
        skipped_processed = 0
        skipped_filter = 0

        for blob in all_blobs:
            if blob.name.endswith("/"):
                continue
            if not blob.name.lower().endswith(Config.AUDIO_EXTENSIONS):
                continue

            filename = os.path.basename(blob.name)

            if blob.name in processed_blobs:
                skipped_processed += 1
                continue

            matches, match_type, match_value = filename_matches_filters(filename, dids, extensions)
            if not matches:
                skipped_filter += 1
                continue

            audio_files.append({
                "blob_name": blob.name,
                "filename": filename,
                "size": blob.size,
                "date_folder": date_folder,
                "match_type": match_type,
                "match_value": match_value,
            })

        log(f"  {date_folder}: {len(audio_files)} new matching files "
            f"(skipped: {skipped_processed} processed, {skipped_filter} filtered, "
            f"{len(all_blobs)} total)", "OK")

        return audio_files

    except Exception as e:
        log(f"Failed listing {date_folder}: {e}", "ERROR")
        traceback.print_exc()
        return []


# =======================================================================
# BATCH API — JSONL GENERATION, SUBMISSION, POLLING, PARSING
# =======================================================================
def build_batch_jsonl_for_file(chunks_meta: List[Dict], original_filename: str,
                               date_folder: str, agent_names: Optional[List[str]] = None) -> Tuple[str, Dict[str, Dict]]:
    """Build JSONL content for batch job from preprocessed audio chunks."""
    total_chunks = len(chunks_meta)
    prompt = get_transcription_prompt(agent_names)
    jsonl_lines = []
    chunk_mapping = {}

    for chunk_info in sorted(chunks_meta, key=lambda x: x["chunk_num"]):
        chunk_num = chunk_info["chunk_num"]
        chunk_path = chunk_info["path"]
        ext = os.path.splitext(chunk_path)[1].lower()
        mime_type = Config.AUDIO_MIME_TYPES.get(ext, "audio/mpeg")

        base_name = os.path.splitext(original_filename)[0]
        gcs_chunk_path = f"{Config.BATCH_CHUNKS_FOLDER}/{date_folder}/{base_name}_chunk{chunk_num:03d}{ext}"
        gcs_uri = upload_to_gcs(chunk_path, Config.OUTPUT_BUCKET, gcs_chunk_path)

        if gcs_uri is None:
            log(f"    Failed to upload chunk {chunk_num} to GCS — skipping", "ERROR")
            continue

        chunk_prompt = prompt
        if total_chunks > 1:
            chunk_prompt += f"\nNOTE: Segment {chunk_num} of {total_chunks}. Transcribe from [00:00]. Complete segment."

        request_id = f"{base_name}_chunk{chunk_num:03d}"

        request_obj = {
            "id": request_id,
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": chunk_prompt},
                            {"fileData": {"fileUri": gcs_uri, "mimeType": mime_type}}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": Config.TEMPERATURE,
                    "maxOutputTokens": Config.TRANSCRIBE_MAX_TOKENS,
                }
            }
        }

        jsonl_lines.append(json.dumps(request_obj, ensure_ascii=False))
        chunk_mapping[request_id] = chunk_info

    jsonl_content = "\n".join(jsonl_lines)
    return jsonl_content, chunk_mapping


def submit_batch_job(client, jsonl_gcs_uri: str, output_gcs_uri: str,
                     job_name: str) -> Optional[Any]:
    """Submit a Gemini batch inference job."""
    try:
        log(f"    Submitting batch job: {job_name}", "BATCH")
        log(f"    Input:  {jsonl_gcs_uri}", "BATCH")
        log(f"    Output: {output_gcs_uri}", "BATCH")

        job = client.batches.create(
            model=Config.GEMINI_MODEL,
            src=jsonl_gcs_uri,
            config=CreateBatchJobConfig(
                dest=output_gcs_uri,
                display_name=job_name,
            ),
        )

        log(f"    Batch job submitted: {job.name}", "BATCH")
        log(f"    Initial state: {job.state}", "BATCH")
        return job
    except Exception as e:
        log(f"    Batch job submission failed: {e}", "ERROR")
        traceback.print_exc()
        return None


def poll_batch_job(client, job_name: str,
                   poll_interval: int = Config.BATCH_POLL_INTERVAL,
                   timeout: int = Config.BATCH_POLL_TIMEOUT) -> Optional[Any]:
    """Poll batch job until completion or timeout."""
    start_time = time.time()
    last_state = None

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            log(f"    Batch job timed out after {format_duration(elapsed)}", "ERROR")
            return None
        if shutdown_requested:
            log("    Shutdown requested — stopping batch poll", "WARN")
            return None

        try:
            job = client.batches.get(name=job_name)
            current_state = str(job.state)
            if current_state != last_state:
                log(f"    Batch job state: {current_state} ({format_duration(elapsed)} elapsed)", "BATCH")
                last_state = current_state

            if job.state == JobState.JOB_STATE_SUCCEEDED:
                log(f"    Batch job SUCCEEDED in {format_duration(elapsed)}", "OK")
                return job
            elif job.state == JobState.JOB_STATE_FAILED:
                log(f"    Batch job FAILED: {getattr(job, 'error', 'unknown error')}", "ERROR")
                return None
            elif job.state == JobState.JOB_STATE_CANCELLED:
                log(f"    Batch job was CANCELLED", "WARN")
                return None
        except Exception as e:
            log(f"    Batch poll error: {e}", "WARN")

        time.sleep(poll_interval)


def parse_batch_results(output_gcs_uri: str, chunk_mapping: Dict[str, Dict]) -> Dict[str, Dict]:
    """Parse batch job results from GCS output JSONL."""
    results = {}

    if output_gcs_uri.startswith("gs://"):
        parts = output_gcs_uri[5:].split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
    else:
        log(f"    Invalid output URI: {output_gcs_uri}", "ERROR")
        return results

    prediction_files = list_gcs_blobs(bucket_name, prefix)
    jsonl_files = [f for f in prediction_files if f.endswith(".jsonl")]

    if not jsonl_files:
        log(f"    No prediction JSONL files found at {output_gcs_uri}", "ERROR")
        return results

    log(f"    Found {len(jsonl_files)} prediction file(s)", "BATCH")

    for jsonl_file in jsonl_files:
        content = download_string_from_gcs(bucket_name, jsonl_file)
        if not content:
            continue

        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"    Invalid JSON in batch output: {e}", "WARN")
                continue

            request_id = entry.get("id", "")
            status = entry.get("status", "")
            if status and "error" in status.lower():
                results[request_id] = {"transcript": None, "status": "failed", "error": status}
                continue

            response = entry.get("response", {})
            candidates = response.get("candidates", [])

            if candidates:
                content_parts = candidates[0].get("content", {}).get("parts", [])
                transcript_text = ""
                for part in content_parts:
                    if part.get("text"):
                        transcript_text += part["text"]

                if transcript_text.strip():
                    results[request_id] = {"transcript": transcript_text.strip(), "status": "success", "error": None}
                else:
                    results[request_id] = {"transcript": None, "status": "failed", "error": "Empty transcript"}
            else:
                results[request_id] = {"transcript": None, "status": "failed", "error": "No candidates"}

    for request_id in chunk_mapping:
        if request_id not in results:
            results[request_id] = {"transcript": None, "status": "failed", "error": "No result from batch API"}

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    log(f"    Parsed results: {success_count}/{len(chunk_mapping)} chunks successful", "BATCH")
    return results


# =======================================================================
# TIMESTAMP HANDLING
# =======================================================================
TIMESTAMP_RE = re.compile(r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]')


def _parse_timestamp(match) -> Optional[int]:
    if isinstance(match, str):
        m = TIMESTAMP_RE.search(match)
        if not m:
            return None
        groups = m.groups()
    else:
        groups = match.groups()
    hours = 0
    minutes = int(groups[0])
    seconds = int(groups[1])
    if groups[2] is not None:
        hours = minutes
        minutes = seconds
        seconds = int(groups[2])
    return hours * 3600 + minutes * 60 + seconds


def _seconds_to_timestamp(seconds: int) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"[{hours}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def adjust_timestamp(line: str, offset_seconds: float) -> str:
    def replacer(match):
        orig_seconds = _parse_timestamp(match)
        if orig_seconds is None:
            return match.group(0)
        return _seconds_to_timestamp(orig_seconds + offset_seconds)
    return TIMESTAMP_RE.sub(replacer, line)


def remap_timestamp(line: str, timestamp_map: Optional[List[Tuple[int, int]]],
                   max_original_sec: Optional[float]) -> str:
    def replacer(match):
        proc_seconds = _parse_timestamp(match)
        if proc_seconds is None:
            return match.group(0)
        proc_ms = proc_seconds * 1000
        orig_ms = _lookup_original_time(timestamp_map, proc_ms)
        if max_original_sec and orig_ms > max_original_sec * 1000:
            orig_ms = max_original_sec * 1000
        return _seconds_to_timestamp(orig_ms / 1000.0)
    return TIMESTAMP_RE.sub(replacer, line)


def _lookup_original_time(timestamp_map: Optional[List[Tuple[int, int]]],
                         processed_ms: int) -> int:
    if not timestamp_map:
        return processed_ms
    if processed_ms <= timestamp_map[0][0]:
        return timestamp_map[0][1]
    if processed_ms >= timestamp_map[-1][0]:
        return timestamp_map[-1][1]
    low, high = 0, len(timestamp_map) - 1
    while low < high - 1:
        mid = (low + high) // 2
        if timestamp_map[mid][0] <= processed_ms:
            low = mid
        else:
            high = mid
    proc1, orig1 = timestamp_map[low]
    proc2, orig2 = timestamp_map[high]
    if proc2 == proc1:
        return orig1
    ratio = (processed_ms - proc1) / (proc2 - proc1)
    return int(orig1 + ratio * (orig2 - orig1))


# =======================================================================
# TRANSCRIPT STITCHING & VALIDATION
# =======================================================================
def combine_chunks(chunk_results: Dict[str, Dict], chunks_meta: List[Dict],
                  timestamp_map: Optional[List[Tuple[int, int]]],
                  original_duration_sec: float) -> Tuple[str, List[str]]:
    """Stitch chunks with deduplication and validation."""
    sorted_chunks = sorted(chunks_meta, key=lambda x: x["chunk_num"])
    final_lines = []
    seen_lines = set()
    failed_chunks = []

    for chunk_info in sorted_chunks:
        chunk_id = f"{os.path.splitext(chunk_info['original_filename'])[0]}_chunk{chunk_info['chunk_num']:03d}"
        result = chunk_results.get(chunk_id, {})

        if result.get("status") != "success" or not result.get("transcript"):
            log(f"    ⚠️ Skipping failed chunk {chunk_info['chunk_num']}", "WARN")
            failed_chunks.append(chunk_id)
            continue

        raw_transcript = result["transcript"].strip()
        cleaned_transcript = clean_chunk_transcript(raw_transcript)

        offset = chunk_info["start_time"]
        for line in cleaned_transcript.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            adjusted = adjust_timestamp(stripped, offset)
            remapped = remap_timestamp(adjusted, timestamp_map, original_duration_sec)
            norm_key = _normalize_text(remapped)
            if norm_key not in seen_lines:
                final_lines.append(remapped)
                seen_lines.add(norm_key)

    if failed_chunks:
        log(f"    ❌ Missing {len(failed_chunks)} chunks: {', '.join(failed_chunks[:3])}" +
            ("..." if len(failed_chunks) > 3 else ""), "ERROR")

    return "\n".join(final_lines), failed_chunks


def calculate_transcript_duration(transcript: str) -> float:
    max_seconds = 0.0
    for match in TIMESTAMP_RE.finditer(transcript):
        ts_seconds = _parse_timestamp(match)
        if ts_seconds is not None:
            max_seconds = max(max_seconds, ts_seconds)
    return max_seconds


# =======================================================================
# FILE PROCESSING
# =======================================================================
def extract_metadata_from_filename(filename: str) -> Dict[str, Optional[str]]:
    metadata = {"call_id": None, "call_date": None, "call_time": None}
    try:
        base = os.path.splitext(os.path.basename(filename))[0]
        cleaned = re.sub(r'^processed[_ ]+', '', base, flags=re.IGNORECASE)

        date_match = re.search(
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            cleaned, re.IGNORECASE
        )
        if date_match:
            metadata["call_date"] = date_match.group(1).strip()
        else:
            date_match2 = re.search(
                r'(\d{1,2})[_ ]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[_ ]+(\d{4})',
                cleaned, re.IGNORECASE
            )
            if date_match2:
                metadata["call_date"] = f"{date_match2.group(1)} {date_match2.group(2)} {date_match2.group(3)}"

        time_match = re.search(r'(\d{1,2})-(\d{2})-(\d{2})(?=-|\s|$)', cleaned)
        if time_match:
            metadata["call_time"] = f"{time_match.group(1)}:{time_match.group(2)}:{time_match.group(3)}"

        remainder = cleaned[(date_match.end() if date_match else 0):]
        parts = remainder.replace(' ', '-').strip('-').split('-')
        for part in reversed(parts):
            if '.' in part and re.match(r'^\d+\.\d+$', part):
                metadata["call_id"] = part
                break
        if not metadata["call_id"]:
            for part in reversed(parts):
                if re.match(r'^\d{5,}', part):
                    metadata["call_id"] = part
                    break
    except Exception as e:
        log(f"Metadata extraction error: {e}", "WARN")
    return metadata


def save_result(save_info: Dict) -> Tuple[str, bool]:
    """Save transcript locally and to GCS."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = os.path.splitext(save_info["audio_filename"])[0]
        orig_dur = format_duration(save_info["original_duration"]) if save_info.get("original_duration") else "?"
        proc_dur = format_duration(save_info["preprocessed_duration"]) if save_info.get("preprocessed_duration") else "?"
        date_folder = save_info["date_folder"]
        metadata = save_info.get("file_metadata", {})

        meta_lines = []
        if metadata.get("call_id"):
            meta_lines.append(f"Call ID: {metadata['call_id']}")
        if metadata.get("call_date"):
            meta_lines.append(f"Call Date: {metadata['call_date']}")
        if metadata.get("call_time"):
            meta_lines.append(f"Call Time: {metadata['call_time']}")
        if date_folder:
            meta_lines.append(f"Date Folder: {date_folder}")
        if save_info.get("match_type"):
            meta_lines.append(f"Match: {save_info['match_type']}={save_info.get('match_value', '')}")
        meta_text = "\n".join(meta_lines) + "\n" if meta_lines else ""

        gpu_info = f"GPU: {DEVICE_INFO['gpu_name']}" if DEVICE == "cuda" else "CPU"
        transcript_text = (
            f"FULL CALL TRANSCRIPT\n{'='*70}\n"
            f"Source: {save_info['audio_filename']}\n"
            f"{meta_text}"
            f"Original Duration: {orig_dur}\n"
            f"Speech Duration: {proc_dur}\n"
            f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Pipeline: Preprocess→Gemini Batch API ({Config.CHUNK_DURATION_MINUTES}min, "
            f"{Config.CHUNK_OVERLAP_SECONDS}s overlap)\n"
            f"Model: {Config.GEMINI_MODEL}\n"
            f"Compute: {gpu_info} (preprocessing) + Batch API (transcription)\n"
            f"Timestamps: Remapped to original\n"
            f"{'='*70}\n\n"
            f"{save_info['full_transcript']}\n"
        )

        local_dir = os.path.join(Config.LOCAL_RESULTS_DIR, date_folder) if date_folder else Config.LOCAL_RESULTS_DIR
        result_filename = f"{base_name}_{timestamp}_transcript.txt"
        local_path = os.path.join(local_dir, result_filename)

        with _save_lock:
            os.makedirs(local_dir, exist_ok=True)

        with open(local_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        log(f"    Saved locally: {local_path}", "SAVE")

        client = get_storage_client()
        gcs_path = f"{Config.OUTPUT_FOLDER}/{date_folder}/{result_filename}" if date_folder else f"{Config.OUTPUT_FOLDER}/{result_filename}"
        bucket = client.bucket(Config.OUTPUT_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(transcript_text, content_type="text/plain; charset=utf-8")
        log(f"    Uploaded: gs://{Config.OUTPUT_BUCKET}/{gcs_path}", "SAVE")

        return save_info["audio_filename"], True
    except Exception as e:
        log(f"  Save error: {e}", "ERROR")
        traceback.print_exc()
        return save_info.get("audio_filename", "?"), False


def preprocess_single_file(file_info: Dict, date_folder: str,
                           file_index: int, total_files: int) -> Optional[Dict]:
    """Phase 1: Download and preprocess a single file."""
    filename = file_info["filename"]
    blob_name = file_info["blob_name"]
    match_type = file_info.get("match_type", "")
    match_value = file_info.get("match_value", "")

    file_hash = hashlib.md5(blob_name.encode()).hexdigest()[:12]
    work_dir = os.path.join("work_files", f"{file_hash}_{threading.current_thread().name}")
    os.makedirs(work_dir, exist_ok=True)

    log(f"  ▶ [{file_index}/{total_files}] {filename} ({match_type}={match_value})", "PROCESSING")

    try:
        log(f"    Step 1: Download...", "DOWNLOAD")
        download_dir = os.path.join(work_dir, "download")
        local_path = download_from_gcs(blob_name, filename, download_dir)
        if not local_path:
            return None
        log(f"    Downloaded: {format_file_size(os.path.getsize(local_path))}", "OK")

        log(f"    Step 2: Preprocess...", "PREPROCESS")
        metadata = extract_metadata_from_filename(filename)
        preprocessed_path, timestamp_map, orig_duration, proc_duration = preprocess_audio(
            local_path, filename, work_dir
        )
        if preprocessed_path is None:
            return None

        log(f"    Step 3: Chunk...", "PROCESSING")
        chunk_dir = os.path.join(work_dir, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        chunks, _ = split_chunks(preprocessed_path, chunk_dir, filename)
        if not chunks or not chunks[0].get("path"):
            return None
        log(f"    {len(chunks)} chunk(s)", "OK")

        return {
            "file_info": file_info, "filename": filename, "blob_name": blob_name,
            "chunks_meta": chunks, "timestamp_map": timestamp_map,
            "orig_duration": orig_duration, "proc_duration": proc_duration,
            "metadata": metadata, "work_dir": work_dir,
            "match_type": match_type, "match_value": match_value,
        }
    except Exception as e:
        log(f"  ❌ [{file_index}/{total_files}] Preprocess error for {filename}: {e}", "ERROR")
        traceback.print_exc()
        return None


def run_batch_transcription_cycle(date_folder: str, preprocessed_files: List[Dict],
                                  vertex_client, tracker,
                                  agent_names: Optional[List[str]] = None) -> Tuple[int, int]:
    """Phase 2: Submit all preprocessed files as a single batch job."""
    if not preprocessed_files:
        return 0, 0

    # =====================================================================
    # Step 1: Upload ALL chunks to GCS in parallel (was sequential = 86 min!)
    # =====================================================================
    log(f"  Uploading audio chunks to GCS for {len(preprocessed_files)} files...", "BATCH")
    upload_start = time.time()

    # Collect all chunks that need uploading
    upload_tasks = []  # list of (chunk_info, gcs_path, original_filename)
    for pf in preprocessed_files:
        filename = pf["filename"]
        base_name = os.path.splitext(filename)[0]
        for chunk_info in pf["chunks_meta"]:
            chunk_num = chunk_info["chunk_num"]
            chunk_path = chunk_info["path"]
            ext = os.path.splitext(chunk_path)[1].lower()
            gcs_chunk_path = f"{Config.BATCH_CHUNKS_FOLDER}/{date_folder}/{base_name}_chunk{chunk_num:03d}{ext}"
            upload_tasks.append((chunk_info, chunk_path, gcs_chunk_path, filename))

    log(f"  {len(upload_tasks)} chunks to upload...", "UPLOAD")

    # Upload in parallel using 10 threads
    chunk_gcs_uris = {}  # (filename, chunk_num) → gcs_uri
    upload_errors = 0

    def _upload_one(task):
        chunk_info, chunk_path, gcs_path, orig_filename = task
        gcs_uri = upload_to_gcs(chunk_path, Config.OUTPUT_BUCKET, gcs_path)
        return (orig_filename, chunk_info["chunk_num"], gcs_uri)

    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="UL") as executor:
        futures = [executor.submit(_upload_one, t) for t in upload_tasks]
        for future in as_completed(futures):
            try:
                orig_filename, chunk_num, gcs_uri = future.result()
                if gcs_uri:
                    chunk_gcs_uris[(orig_filename, chunk_num)] = gcs_uri
                else:
                    upload_errors += 1
            except Exception as e:
                upload_errors += 1
                log(f"    Chunk upload exception: {e}", "ERROR")

    upload_elapsed = time.time() - upload_start
    log(f"  Uploaded {len(chunk_gcs_uris)}/{len(upload_tasks)} chunks to GCS "
        f"({format_duration(upload_elapsed)}, {upload_errors} errors)", "UPLOAD")

    if not chunk_gcs_uris:
        log("  No chunks uploaded — all failed", "ERROR")
        return 0, len(preprocessed_files)

    # =====================================================================
    # Step 2: Build JSONL from uploaded chunk URIs
    # =====================================================================
    log(f"  Building batch JSONL...", "BATCH")
    prompt = get_transcription_prompt(agent_names)
    all_jsonl_lines = []
    all_chunk_mappings = {}
    file_chunk_registry = {}

    for pf in preprocessed_files:
        filename = pf["filename"]
        base_name = os.path.splitext(filename)[0]
        total_chunks = len(pf["chunks_meta"])
        file_req_ids = []

        for chunk_info in sorted(pf["chunks_meta"], key=lambda x: x["chunk_num"]):
            chunk_num = chunk_info["chunk_num"]
            gcs_uri = chunk_gcs_uris.get((filename, chunk_num))
            if not gcs_uri:
                continue

            ext = os.path.splitext(chunk_info["path"])[1].lower()
            mime_type = Config.AUDIO_MIME_TYPES.get(ext, "audio/mpeg")

            chunk_prompt = prompt
            if total_chunks > 1:
                chunk_prompt += f"\nNOTE: Segment {chunk_num} of {total_chunks}. Transcribe from [00:00]. Complete segment."

            request_id = f"{base_name}_chunk{chunk_num:03d}"

            request_obj = {
                "id": request_id,
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": chunk_prompt},
                                {"fileData": {"fileUri": gcs_uri, "mimeType": mime_type}}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": Config.TEMPERATURE,
                        "maxOutputTokens": Config.TRANSCRIBE_MAX_TOKENS,
                    }
                }
            }

            all_jsonl_lines.append(json.dumps(request_obj, ensure_ascii=False))
            all_chunk_mappings[request_id] = chunk_info
            file_req_ids.append(request_id)

        if file_req_ids:
            file_chunk_registry[filename] = file_req_ids

    if not all_jsonl_lines:
        log("  No JSONL lines built — all chunk uploads failed", "ERROR")
        return 0, len(preprocessed_files)

    combined_jsonl = "\n".join(all_jsonl_lines)
    total_chunks = len(all_chunk_mappings)
    log(f"  Total chunks in batch: {total_chunks} across {len(file_chunk_registry)} files", "BATCH")

    # =====================================================================
    # Step 3: Upload JSONL to GCS
    # =====================================================================
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_gcs_path = f"{Config.BATCH_INPUT_FOLDER}/{date_folder}/batch_{batch_timestamp}.jsonl"
    jsonl_gcs_uri = upload_string_to_gcs(combined_jsonl, Config.OUTPUT_BUCKET, jsonl_gcs_path)
    if not jsonl_gcs_uri:
        log("  Failed to upload batch JSONL to GCS", "ERROR")
        return 0, len(preprocessed_files)
    log(f"  Uploaded batch JSONL: {jsonl_gcs_uri} ({len(combined_jsonl)} bytes)", "UPLOAD")

    # =====================================================================
    # Step 4: Submit batch job
    # =====================================================================
    output_gcs_uri = f"gs://{Config.OUTPUT_BUCKET}/{Config.BATCH_OUTPUT_FOLDER}/{date_folder}/batch_{batch_timestamp}/"
    job_name_display = f"transcribe_{date_folder}_{batch_timestamp}"

    job = submit_batch_job(vertex_client, jsonl_gcs_uri, output_gcs_uri, job_name_display)
    if job is None:
        log("  Batch job submission failed", "ERROR")
        return 0, len(preprocessed_files)

    # =====================================================================
    # Step 5: Poll until complete
    # =====================================================================
    log(f"  Polling batch job (interval={Config.BATCH_POLL_INTERVAL}s, "
        f"timeout={format_duration(Config.BATCH_POLL_TIMEOUT)})...", "BATCH")
    completed_job = poll_batch_job(vertex_client, job.name)

    if completed_job is None:
        log("  Batch job did not complete in time — files left untracked for retry", "ERROR")
        log(f"  Batch job name: {job.name}", "ERROR")
        log(f"  Check status: https://console.cloud.google.com/vertex-ai/batch-predictions", "ERROR")
        log(f"  Re-run the same --date to retry (unprocessed files will be picked up)", "ERROR")
        # DON'T mark as failed — leave untracked so re-running picks them up
        return 0, len(preprocessed_files)

    # =====================================================================
    # Step 6: Parse results
    # =====================================================================
    log(f"  Parsing batch results from {output_gcs_uri}...", "BATCH")
    chunk_results = parse_batch_results(output_gcs_uri, all_chunk_mappings)

    # =====================================================================
    # Step 7: Stitch per file and save
    # =====================================================================
    success_count = 0
    failed_count = 0

    for pf in preprocessed_files:
        filename = pf["filename"]
        blob_name = pf["blob_name"]

        try:
            file_results = {}
            for req_id in file_chunk_registry.get(filename, []):
                if req_id in chunk_results:
                    file_results[req_id] = chunk_results[req_id]

            chunks_ok = sum(1 for r in file_results.values() if r.get("status") == "success")
            chunks_fail = len(file_results) - chunks_ok
            log(f"  File {filename}: {chunks_ok}/{len(file_results)} chunks OK", "BATCH")

            full_transcript, failed_chunks = combine_chunks(
                file_results, pf["chunks_meta"], pf["timestamp_map"], pf["orig_duration"]
            )

            if not full_transcript or len(full_transcript) < 50:
                tracker.mark_processed(blob_name, filename, "failed", {"reason": "Transcript too short"})
                failed_count += 1
                continue

            transcript_duration = calculate_transcript_duration(full_transcript)
            duration_ratio = transcript_duration / pf["orig_duration"] if pf["orig_duration"] > 0 else 0

            if duration_ratio < 0.85:
                log(f"    ⚠️ Transcript duration ({transcript_duration:.1f}s) is only "
                    f"{duration_ratio*100:.0f}% of audio ({pf['orig_duration']:.1f}s)", "WARN")
                if duration_ratio < 0.5:
                    tracker.mark_processed(blob_name, filename, "failed", {
                        "reason": f"Severe duration mismatch ({transcript_duration:.1f}/{pf['orig_duration']:.1f}s)"
                    })
                    failed_count += 1
                    continue

            save_info = {
                "full_transcript": full_transcript, "audio_filename": filename,
                "batch_id": "batch_api", "original_duration": pf["orig_duration"],
                "preprocessed_duration": pf["proc_duration"], "file_metadata": pf["metadata"],
                "date_folder": date_folder, "match_type": pf["match_type"],
                "match_value": pf["match_value"],
            }
            _, saved = save_result(save_info)

            if saved:
                tracker.mark_processed(blob_name, filename, "success", {
                    "chars": len(full_transcript), "chunks_ok": chunks_ok,
                    "chunks_fail": chunks_fail, "transcript_duration": transcript_duration,
                    "audio_duration": pf["orig_duration"],
                })
                log(f"  ✅ {filename}: {len(full_transcript)} chars, "
                    f"{chunks_ok}/{len(file_results)} chunks", "DONE")
                success_count += 1
            else:
                tracker.mark_processed(blob_name, filename, "failed", {"reason": "Save failed"})
                failed_count += 1

        except Exception as e:
            log(f"  ❌ Error stitching {filename}: {e}", "ERROR")
            traceback.print_exc()
            tracker.mark_processed(blob_name, filename, "failed",
                                   {"reason": f"Stitch exception: {str(e)[:200]}"})
            failed_count += 1

    # Cleanup
    for pf in preprocessed_files:
        try:
            if pf.get("work_dir"):
                shutil.rmtree(pf["work_dir"], ignore_errors=True)
        except Exception:
            pass

    return success_count, failed_count


# =======================================================================
# TRACKING
# =======================================================================
class ProcessedTracker:
    """Thread-safe tracker for processed files."""
    def __init__(self, filepath: str = Config.PROCESSED_TRACKER_FILE):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._processed: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                self._processed = data.get("processed", {})
                log(f"Loaded tracker: {len(self._processed)} previously processed files", "TRACK")
            except Exception as e:
                log(f"Tracker load error (starting fresh): {e}", "WARN")
                self._processed = {}
        else:
            log("No tracker file found — starting fresh", "TRACK")

    def _save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump({
                    "processed": self._processed,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_processed": len(self._processed)
                }, f, indent=2)
        except Exception as e:
            log(f"Tracker save error: {e}", "ERROR")

    def is_processed(self, blob_name: str) -> bool:
        with self._lock:
            return blob_name in self._processed

    def mark_processed(self, blob_name: str, filename: str, status: str, details: Optional[Dict] = None):
        with self._lock:
            self._processed[blob_name] = {
                "filename": filename, "status": status,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **(details or {})
            }
            self._save()

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            total = len(self._processed)
            success = sum(1 for v in self._processed.values() if v.get("status") == "success")
            failed = sum(1 for v in self._processed.values() if v.get("status") == "failed")
            return {"total": total, "success": success, "failed": failed}

    def get_processed_blobs(self) -> Set[str]:
        with self._lock:
            return set(self._processed.keys())


# =======================================================================
# MAIN EXECUTION — Process date folders and exit
# =======================================================================
def process_date_folder(date_folder: str, dids: List[str], extensions: List[str],
                        tracker: ProcessedTracker, vertex_client, num_workers: int,
                        agent_names: Optional[List[str]] = None) -> Tuple[int, int, int]:
    """
    Process all matching audio files in a single date folder.

    Phase 1: Preprocess all files (parallel GPU/CPU)
    Phase 2: Submit one batch job for ALL chunks
    Phase 3: Poll, parse, stitch, save
    """
    log(f"{'='*75}", "HOUR")
    log(f"PROCESSING DATE FOLDER: {date_folder}", "HOUR")
    log(f"{'='*75}", "HOUR")

    new_files = list_audio_files_in_date_folder(date_folder, dids, extensions, tracker)

    if not new_files:
        log(f"  No new matching files in {date_folder}", "OK")
        return 0, 0, 0

    total = len(new_files)
    effective_workers = min(num_workers, total)
    log(f"  Found {total} new files → preprocessing with {effective_workers} workers...", "MEGA")

    # Phase 1: Preprocess
    log(f"  === PHASE 1: PREPROCESSING ({total} files) ===", "BATCH")
    preprocess_start = time.time()
    preprocessed_files = []

    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="PP") as executor:
        future_to_info = {}
        for idx, file_info in enumerate(new_files, 1):
            future = executor.submit(preprocess_single_file, file_info, date_folder, idx, total)
            future_to_info[future] = (idx, file_info)

        for future in as_completed(future_to_info):
            idx, file_info = future_to_info[future]
            try:
                result = future.result()
                if result:
                    preprocessed_files.append(result)
                else:
                    tracker.mark_processed(file_info["blob_name"], file_info["filename"],
                                           "failed", {"reason": "Preprocessing failed"})
            except Exception as e:
                tracker.mark_processed(file_info["blob_name"], file_info["filename"],
                                       "failed", {"reason": f"Preprocess exception: {str(e)[:200]}"})

    preprocess_elapsed = time.time() - preprocess_start
    log(f"  Preprocessing done: {len(preprocessed_files)}/{total} files "
        f"({format_duration(preprocess_elapsed)})", "OK")

    if not preprocessed_files:
        log("  No files preprocessed successfully — skipping batch job", "WARN")
        return 0, total, total

    # Phase 2: Batch transcription
    log(f"  === PHASE 2: BATCH TRANSCRIPTION ({len(preprocessed_files)} files) ===", "BATCH")
    batch_start = time.time()

    success, failed = run_batch_transcription_cycle(
        date_folder, preprocessed_files, vertex_client, tracker, agent_names
    )

    batch_elapsed = time.time() - batch_start
    preprocess_failed = total - len(preprocessed_files)
    total_failed = failed + preprocess_failed

    stats = tracker.get_stats()
    log(f"  Date {date_folder} done: {success}✅ {total_failed}❌ / {total} "
        f"(preprocess: {format_duration(preprocess_elapsed)}, "
        f"batch: {format_duration(batch_elapsed)})", "DONE")
    log(f"  Running total: {stats['success']}✅ {stats['failed']}❌ / {stats['total']} processed", "TRACK")

    return success, total_failed, total


def run_processing(date_folders: List[str], num_workers: int = Config.DEFAULT_WORKERS):
    """Process one or more date folders sequentially, then exit."""
    dids = load_filter_list(Config.DIDS_FILE, "DIDs")
    extensions = load_filter_list(Config.EXTENSIONS_FILE, "Extensions")

    if not dids and not extensions:
        log("No DIDs or Extensions loaded — nothing to filter by. "
            "Create dids.txt and/or extensions.txt with numbers, one per line.", "ERROR")
        return

    agent_names = load_agent_names(Config.AGENT_NAMES_FILE)
    tracker = ProcessedTracker()

    gpu_info = f"🎮 {DEVICE_INFO['gpu_name']} ({DEVICE_INFO['vram_total_gb']}GB)" if DEVICE == "cuda" else "⚠️ CPU mode"

    log("=" * 75, "INFO")
    log(f"  📦 BATCH API TRANSCRIPTION PIPELINE", "INFO")
    log(f"  {gpu_info}", "INFO")
    log(f"  Workers: {num_workers} (preprocessing) | Transcription: Gemini Batch API", "INFO")
    log(f"  Model: {Config.GEMINI_MODEL} | {Config.CHUNK_DURATION_MINUTES}min chunks", "INFO")
    log(f"  Batch poll: every {Config.BATCH_POLL_INTERVAL}s, timeout {format_duration(Config.BATCH_POLL_TIMEOUT)}", "INFO")
    log(f"  Cost: 50% of sequential API (batch pricing)", "INFO")
    log(f"  Date folders: {', '.join(date_folders)}", "INFO")
    log(f"  DIDs: {len(dids)} | Extensions: {len(extensions)}", "INFO")
    log(f"  Agent names: {len(agent_names)}" if agent_names else "  Agent names: none", "INFO")
    log(f"  Tracker: {tracker.filepath}", "INFO")
    log("=" * 75, "INFO")

    if not _verify_setup():
        return

    vertex_client = get_vertex_client()
    log("Pre-loading segmenter...", "PREPROCESS")
    get_segmenter()
    if DEVICE == "cuda":
        print_gpu_status()

    os.makedirs("work_files", exist_ok=True)

    grand_success = 0
    grand_failed = 0
    grand_total = 0
    pipeline_start = time.time()

    for i, date_folder in enumerate(date_folders, 1):
        if shutdown_requested:
            log("Shutdown requested — stopping before next date folder", "WARN")
            break

        log(f"\n{'█'*75}", "HOUR")
        log(f"DATE FOLDER {i}/{len(date_folders)}: {date_folder}", "HOUR")

        success, failed, total = process_date_folder(
            date_folder, dids, extensions, tracker, vertex_client, num_workers, agent_names
        )
        grand_success += success
        grand_failed += failed
        grand_total += total

        if DEVICE == "cuda":
            clear_gpu_cache()

    pipeline_elapsed = time.time() - pipeline_start
    stats = tracker.get_stats()

    log("\n" + "=" * 75, "INFO")
    log("  📊 FINAL SUMMARY", "INFO")
    log("=" * 75, "INFO")
    log(f"  Date folders processed: {len(date_folders)}", "INFO")
    log(f"  This session: {grand_success}✅ {grand_failed}❌ / {grand_total} files", "INFO")
    log(f"  All-time total: {stats['success']}✅ {stats['failed']}❌ / {stats['total']} processed", "INFO")
    log(f"  Total time: {format_duration(pipeline_elapsed)}", "INFO")
    log(f"  Mode: Gemini Batch API (50% cost savings)", "INFO")
    for df in date_folders:
        log(f"  Results: gs://{Config.OUTPUT_BUCKET}/{Config.OUTPUT_FOLDER}/{df}/", "INFO")
    log(f"  Tracker: {tracker.filepath}", "INFO")
    log("=" * 75, "INFO")

    try:
        shutil.rmtree("work_files", ignore_errors=True)
    except Exception:
        pass


# =======================================================================
# SETUP VERIFICATION
# =======================================================================
def _verify_ffmpeg() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        if result.returncode == 0:
            log("ffmpeg OK", "OK")
            return True
    except Exception:
        pass
    log("ffmpeg not found!", "ERROR")
    return False


def _verify_gpu() -> bool:
    if DEVICE == "cuda":
        log(f"GPU: {DEVICE_INFO['gpu_name']} | CUDA {DEVICE_INFO['cuda_version']} | "
            f"{DEVICE_INFO['vram_total_gb']}GB", "GPU")
    else:
        log("No GPU — CPU mode (preprocessing slower, batch API unaffected)", "WARN")
    return True


def _verify_vertex() -> bool:
    try:
        client = get_vertex_client()
        resp = client.models.generate_content(model=Config.GEMINI_MODEL, contents="Say OK")
        if resp and resp.text:
            log(f"Vertex AI OK ({resp.text[:30].strip()})", "OK")
            return True
        return True
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "403" in error_str:
            log(f"Vertex AI AUTH FAILED: {e}", "ERROR")
            return False
        if "404" in error_str:
            log(f"Model not found: {Config.GEMINI_MODEL}", "ERROR")
            return False
        log(f"Vertex AI OK (non-auth error: {error_str[:80]})", "WARN")
        return True


def _verify_gcs() -> bool:
    try:
        client = get_storage_client()
        for bucket_name in [Config.INPUT_BUCKET, Config.OUTPUT_BUCKET]:
            if not client.bucket(bucket_name).exists():
                log(f"Bucket missing: {bucket_name}", "ERROR")
                return False
            log(f"  Bucket OK: gs://{bucket_name}", "OK")
        log("All buckets OK", "OK")
        return True
    except Exception as e:
        log(f"GCS: {e}", "ERROR")
        return False


def _verify_preprocess() -> bool:
    try:
        _ = AudioSegment.silent(duration=100)
        _ = webrtcvad.Vad(Config.VAD_AGGRESSIVENESS)
        log("Preprocessing OK", "OK")
        return True
    except Exception as e:
        log(f"Preprocess: {e}", "ERROR")
        return False


def _verify_setup() -> bool:
    log("Verifying setup...", "PROCESSING")
    checks = [
        ("GPU", _verify_gpu, False),
        ("ffmpeg", _verify_ffmpeg, True),
        ("Vertex AI", _verify_vertex, True),
        ("GCS", _verify_gcs, True),
        ("Preprocessing", _verify_preprocess, True)
    ]
    failed = False
    for name, func, critical in checks:
        try:
            if not func() and critical:
                failed = True
        except Exception as e:
            if critical:
                failed = True
                log(f"  {name}: {e}", "ERROR")
    if failed:
        log("Setup failed", "ERROR")
        return False
    return True


def check_results():
    """Check output bucket for results."""
    log(f"\n📊 Results in gs://{Config.OUTPUT_BUCKET}/{Config.OUTPUT_FOLDER}/", "INFO")
    try:
        client = get_storage_client()
        blobs = list(client.list_blobs(Config.OUTPUT_BUCKET, prefix=f"{Config.OUTPUT_FOLDER}/"))
        result_blobs = [b for b in blobs if not b.name.endswith("/")]
        if not result_blobs:
            log("  (none)", "INFO")
            return
        by_date = {}
        for blob in result_blobs:
            parts = blob.name.split("/")
            date_folder = parts[1] if len(parts) >= 3 else "(root)"
            by_date.setdefault(date_folder, []).append(blob)
        for date_key in sorted(by_date):
            blobs_in_date = by_date[date_key]
            log(f"\n  📅 {date_key} ({len(blobs_in_date)} files)", "INFO")
            for blob in sorted(blobs_in_date, key=lambda x: x.name)[:10]:
                log(f"    • {os.path.basename(blob.name)} ({format_file_size(blob.size)})", "INFO")
    except Exception as e:
        log(f"  Error: {e}", "ERROR")


# =======================================================================
# SIGNAL HANDLING
# =======================================================================
def signal_handler(sig, frame):
    global shutdown_requested
    log("Shutdown requested via signal. Finishing current processing...", "WARN")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)


# =======================================================================
# CLI
# =======================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch API Vertex AI Transcription Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process a single date folder
  python transcript.py --date 2026-02-28

  # Process multiple date folders sequentially
  python transcript.py --date 2026-02-28 2026-03-01

  # With 3 preprocessing workers
  python transcript.py --date 2026-02-28 --workers 3

  # List available date folders
  python transcript.py --list

  # Check results
  python transcript.py --check

FOLDER STRUCTURE:
  gs://travel-audio-batch-input/teleappliant_recordings/2026-02-28/*.mp3
  (recordings sit directly in date folder, no subfolders)

HARDWARE:
  8GB GPU is sufficient. inaSpeechSegmenter uses ~1-2GB VRAM.
"""
    )
    parser.add_argument("--date", type=str, nargs='+', required=False,
                       help="Date folder(s) to process (YYYY-MM-DD). Can specify multiple.")
    parser.add_argument("--workers", type=int, default=Config.DEFAULT_WORKERS,
                       help=f"Parallel preprocessing workers (default: {Config.DEFAULT_WORKERS})")
    parser.add_argument("--list", action="store_true",
                       help="List date folders in input bucket")
    parser.add_argument("--check", action="store_true",
                       help="Check results in output bucket")
    parser.add_argument("--status", action="store_true",
                       help="Show tracker status")
    parser.add_argument("--reset-tracker", action="store_true",
                       help="Reset the processed files tracker")
    parser.add_argument("--gpu-info", action="store_true",
                       help="Show GPU info")
    parser.add_argument("--dids", type=str, default=Config.DIDS_FILE,
                       help=f"DIDs file (default: {Config.DIDS_FILE})")
    parser.add_argument("--extensions", type=str, default=Config.EXTENSIONS_FILE,
                       help=f"Extensions file (default: {Config.EXTENSIONS_FILE})")
    parser.add_argument("--agent-names", type=str, default=Config.AGENT_NAMES_FILE,
                       help=f"Agent names file (default: {Config.AGENT_NAMES_FILE})")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon (background process)")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(daemon_mode=args.daemon)

    if args.daemon:
        log("Starting in daemon mode...", "INFO")
        if not become_daemon():
            log("Failed to daemonize - exiting", "ERROR")
            sys.exit(1)
        log("Daemon process started successfully", "OK")

    if args.dids != Config.DIDS_FILE:
        Config.DIDS_FILE = args.dids
    if args.extensions != Config.EXTENSIONS_FILE:
        Config.EXTENSIONS_FILE = args.extensions
    if args.agent_names != Config.AGENT_NAMES_FILE:
        Config.AGENT_NAMES_FILE = args.agent_names

    if args.gpu_info:
        log(f"\n🎮 GPU: {DEVICE_INFO['device'].upper()}", "INFO")
        if DEVICE == "cuda":
            log(f"  {DEVICE_INFO['gpu_name']} | CUDA {DEVICE_INFO['cuda_version']} | {DEVICE_INFO['vram_total_gb']}GB", "INFO")
        else:
            log("  No GPU (CPU mode)", "INFO")
        log(f"  PyTorch: {torch.__version__}", "INFO")
        log(f"  Note: inaSpeechSegmenter needs ~1-2GB VRAM. 8GB GPU is sufficient.", "INFO")
        return

    if args.list:
        try:
            client = get_storage_client()
            iterator = client.list_blobs(Config.INPUT_BUCKET,
                                        prefix=f"{Config.INPUT_BASE_FOLDER}/", delimiter="/")
            _ = list(iterator)
            for prefix in sorted(iterator.prefixes):
                folder_name = prefix.rstrip("/").split("/")[-1]
                # Count files in folder
                file_blobs = list(client.list_blobs(Config.INPUT_BUCKET, prefix=prefix))
                audio_count = sum(1 for b in file_blobs
                                 if not b.name.endswith("/") and
                                 b.name.lower().endswith(Config.AUDIO_EXTENSIONS))
                log(f"  📅 {folder_name} — {audio_count} audio files", "INFO")
        except Exception as e:
            log(f"  Error: {e}", "ERROR")
        return

    if args.check:
        check_results()
        return

    if args.status:
        tracker = ProcessedTracker()
        stats = tracker.get_stats()
        log(f"\n📋 Tracker: {Config.PROCESSED_TRACKER_FILE}", "INFO")
        log(f"  Total processed: {stats['total']}", "INFO")
        log(f"  Success: {stats['success']}", "INFO")
        log(f"  Failed: {stats['failed']}", "INFO")
        return

    if args.reset_tracker:
        if os.path.exists(Config.PROCESSED_TRACKER_FILE):
            os.remove(Config.PROCESSED_TRACKER_FILE)
            log(f"✅ Tracker reset: {Config.PROCESSED_TRACKER_FILE} deleted", "OK")
        else:
            log(f"ℹ️ No tracker file to reset", "INFO")
        return

    if not args.date:
        log("ERROR: --date is required. Use --date YYYY-MM-DD [YYYY-MM-DD ...]", "ERROR")
        log("       Use --list to see available date folders.", "INFO")
        return

    # Validate all date formats
    date_folders = []
    for d in args.date:
        try:
            datetime.strptime(d, "%Y-%m-%d")
            date_folders.append(d)
        except ValueError:
            log(f"Bad date format: {d} (use YYYY-MM-DD) — skipping", "ERROR")

    if not date_folders:
        log("No valid date folders to process", "ERROR")
        return

    workers = min(args.workers, Config.MAX_WORKERS)
    run_processing(date_folders, num_workers=workers)


if __name__ == "__main__":
    try:
        from google import genai
        from google.genai.types import HttpOptions, CreateBatchJobConfig, JobState
    except ImportError:
        log("ERROR: google-genai not installed. Run: pip install google-genai", "ERROR")
        sys.exit(1)

    try:
        from google.cloud import storage
    except ImportError:
        log("ERROR: google-cloud-storage not installed. Run: pip install google-cloud-storage", "ERROR")
        sys.exit(1)

    try:
        from pydub import AudioSegment
    except ImportError:
        log("ERROR: pydub not installed. Run: pip install pydub", "ERROR")
        sys.exit(1)

    try:
        import webrtcvad
    except ImportError:
        log("ERROR: webrtcvad not installed. Run: pip install webrtcvad", "ERROR")
        sys.exit(1)

    try:
        from inaSpeechSegmenter import Segmenter
    except ImportError:
        log("ERROR: inaSpeechSegmenter not installed. Run: pip install inaSpeechSegmenter", "ERROR")
        sys.exit(1)

    main()
