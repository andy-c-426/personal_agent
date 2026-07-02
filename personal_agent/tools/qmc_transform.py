"""QQ Music encrypted file decryption and Apple Music M4A transcoding.

Decrypts QMC-encrypted files (.qmcflac, .qmcogg, .qmcmp3) using the known
static XOR cipher, and map-cipher files (.mflac, .mgg) using embedded-key
decryption. Optionally transcodes to M4A via ffmpeg for Apple Music.
"""

import json
import logging
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QMC PRNG key generation
# ---------------------------------------------------------------------------


def _prng_key(seed: int, length: int = 256) -> bytes:
    """Generate a key using the QMC LCG PRNG: seed = (seed * 191 + 13) & 0xFF."""
    key = bytearray(length)
    for i in range(length):
        seed = (seed * 191 + 13) & 0xFF
        key[i] = seed
    return bytes(key)


# Pre-computed 256-byte static XOR key for QMC v1 files.
# Generated with _prng_key(0) — this matches the well-known default key
# used for .qmcflac / .qmcogg / .qmcmp3 decryption.
_QMC_V1_KEY: bytes = _prng_key(0)

# Pre-computed map-cipher seed table used to decrypt the per-file key
# embedded in .mflac / .mgg footers.
_QMC_MAP_SEED: bytes = _prng_key(1)

# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

_QMC_V1_EXTENSIONS: frozenset[str] = frozenset({".qmcflac", ".qmcogg", ".qmcmp3"})
_MAP_CIPHER_EXTENSIONS: frozenset[str] = frozenset({".mflac", ".mgg"})
_ALL_QMC_EXTENSIONS: frozenset[str] = _QMC_V1_EXTENSIONS | _MAP_CIPHER_EXTENSIONS
_PLAIN_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".flac", ".ogg", ".mp3", ".wav", ".wma", ".m4a"})

_EXT_TO_FORMAT: dict[str, str] = {
    ".flac": "flac", ".ogg": "ogg", ".mp3": "mp3",
    ".wav": "wav", ".wma": "wma", ".m4a": "mp3",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR *data* with the repeating *key*."""
    key_len = len(key)
    return bytes(b ^ key[i % key_len] for i, b in enumerate(data))


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# (magic, offset, format_name)
_MAGIC_SIGNATURES: list[tuple[bytes, int, str]] = [
    (b"fLaC", 0, "flac"),
    (b"OggS", 0, "ogg"),
    (b"\xff\xfb", 0, "mp3"),
    (b"\xff\xf3", 0, "mp3"),
    (b"\xff\xf2", 0, "mp3"),
    (b"\xff\xfa", 0, "mp3"),
    (b"ID3", 0, "mp3"),
    (b"\x30\x26\xb2\x75", 4, "wma"),
]


def detect_format(data: bytes) -> str:
    """Detect audio codec from magic bytes. Returns 'flac', 'mp3', 'ogg', 'wma', or 'unknown'."""
    for magic, offset, fmt in _MAGIC_SIGNATURES:
        end = offset + len(magic)
        if len(data) >= end and data[offset:end] == magic:
            return fmt
    return "unknown"


# ---------------------------------------------------------------------------
# QMC v1 decryption (static key)
# ---------------------------------------------------------------------------


def _decrypt_qmc_v1(data: bytes) -> bytes:
    """Decrypt QMC v1 data using the static 256-byte key."""
    return _xor_bytes(data, _QMC_V1_KEY)


# ---------------------------------------------------------------------------
# Map cipher decryption (.mflac, .mgg)
# ---------------------------------------------------------------------------


def _decrypt_map(data: bytes) -> bytes:
    """Decrypt a map-cipher file.

    Map-cipher files embed a per-file decryption key near the end of the
    file.  The key itself is XOR-encrypted with the static map seed table.
    The audio data (everything before the key material) is XOR-decrypted
    with that derived key.
    """
    size = len(data)
    if size < 10:
        raise ValueError("File too small for map-cipher decryption")

    # The footer is at the end of the file: [key_data][footer_marker]
    # Footer marker: 4 bytes key_length (LE), then optional 1-byte flag.
    # The key material sits right before the footer marker.

    # Try reading the key length from the last 4 bytes.
    key_len = struct.unpack_from("<I", data, size - 4)[0]

    if key_len == 0 or key_len > size - 8:
        raise ValueError("Could not parse map-cipher footer (invalid key length)")

    key_start = size - 4 - key_len
    encrypted_key = data[key_start : key_start + key_len]

    # Decrypt the embedded key with the static map seed.
    decrypted_key = _xor_bytes(encrypted_key, _QMC_MAP_SEED)

    # Decrypt the audio data (everything before the key material).
    audio_data = data[:key_start]
    return _xor_bytes(audio_data, decrypted_key)


# ---------------------------------------------------------------------------
# Main decryption dispatcher
# ---------------------------------------------------------------------------


def decrypt_qmc(filepath: Path) -> bytes:
    """Decrypt a QMC-encrypted audio file.

    Returns the raw decrypted audio bytes.  Supports:
      - .qmcflac, .qmcogg, .qmcmp3  (static XOR key)
      - .mflac, .mgg                 (map cipher with embedded key)

    Raises:
        FileNotFoundError: *filepath* does not exist.
        ValueError: Unsupported extension or corrupt file.
    """
    path = filepath.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    data = path.read_bytes()
    if len(data) == 0:
        raise ValueError("File is empty")

    ext = path.suffix.lower()
    if ext in _QMC_V1_EXTENSIONS:
        return _decrypt_qmc_v1(data)
    if ext in _MAP_CIPHER_EXTENSIONS:
        return _decrypt_map(data)

    raise ValueError(
        f"Unsupported format: {ext}. "
        f"Supported: {', '.join(sorted(_ALL_QMC_EXTENSIONS))}"
    )


# ---------------------------------------------------------------------------
# ffmpeg transcoding
# ---------------------------------------------------------------------------

_CODEC_ARGS: dict[str, list[str]] = {
    # lossless
    "flac": ["-acodec", "alac", "-movflags", "+faststart"],
    "wav": ["-acodec", "alac", "-movflags", "+faststart"],
    # lossy
    "mp3": ["-acodec", "aac", "-b:a", "256k", "-movflags", "+faststart"],
    "ogg": ["-acodec", "aac", "-b:a", "256k", "-movflags", "+faststart"],
    "wma": ["-acodec", "aac", "-b:a", "256k", "-movflags", "+faststart"],
    # fallback
    "unknown": ["-acodec", "aac", "-b:a", "256k", "-movflags", "+faststart"],
}


def transcode_to_m4a(
    input_path: Path,
    output_path: Path,
    audio_format: str = "unknown",
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """Transcode a decrypted audio file to M4A using ffmpeg.

    Uses ALAC for lossless sources (FLAC, WAV) and AAC for lossy ones.

    Raises:
        RuntimeError: ffmpeg binary not found.
        subprocess.CalledProcessError: ffmpeg invocation failed.
    """
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(
            f"ffmpeg not found. Install it: brew install ffmpeg"
        )

    args = _CODEC_ARGS.get(audio_format, _CODEC_ARGS["unknown"])
    cmd = [
        ffmpeg_bin,
        "-y",  # overwrite output
        "-i",
        str(input_path),
        *args,
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Agent tool function
# ---------------------------------------------------------------------------


def transform_qmc(
    filepath: str,
    output_dir: Optional[str] = None,
) -> str:
    """Transform an audio file to Apple Music-compatible M4A.

    Handles two kinds of input:

    1. **QQ Music encrypted files** (.qmcflac, .qmcogg, .qmcmp3, .mflac,
       .mgg) — decrypted using the known cipher, then transcoded.
    2. **Plain audio files** (.flac, .ogg, .mp3, .wav, .wma) — transcoded
       directly without decryption.

    Transcoding uses ALAC for lossless sources (FLAC, WAV) and AAC for
    lossy ones (MP3, OGG).  Requires ffmpeg.

    Args:
        filepath: Path to the audio file.
        output_dir: Directory for the output M4A.  Defaults to the same
                    directory as the input file.

    Returns:
        JSON string with ``{"status": "success", ...}`` or
        ``{"status": "error", "message": "..."}``.
    """
    try:
        input_path = Path(filepath).expanduser().resolve()
        ext = input_path.suffix.lower()

        # Determine output directory
        if output_dir:
            out_dir = Path(output_dir).expanduser().resolve()
        else:
            out_dir = input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        output_path = out_dir / f"{input_path.stem}.m4a"

        # --- Path 1: QQ Music encrypted ---
        if ext in _ALL_QMC_EXTENSIONS:
            decrypted = decrypt_qmc(input_path)
            fmt = detect_format(decrypted)
            decrypted_path = out_dir / f"{input_path.stem}_decrypted.{_fmt_to_ext(fmt)}"
            decrypted_path.write_bytes(decrypted)
            try:
                transcode_to_m4a(decrypted_path, output_path, audio_format=fmt)
            finally:
                decrypted_path.unlink(missing_ok=True)

        # --- Path 2: Plain audio ---
        elif ext in _PLAIN_AUDIO_EXTENSIONS:
            fmt = _EXT_TO_FORMAT.get(ext, "unknown")
            transcode_to_m4a(input_path, output_path, audio_format=fmt)

        else:
            supported = sorted(_ALL_QMC_EXTENSIONS | _PLAIN_AUDIO_EXTENSIONS)
            return json.dumps(
                {"status": "error",
                 "message": f"Unsupported format: {ext}. Supported: {', '.join(supported)}"},
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "status": "success",
                "output_path": str(output_path),
                "format": fmt,
                "original": str(input_path),
            },
            ensure_ascii=False,
        )
    except FileNotFoundError as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
    except RuntimeError as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else str(e)
        return json.dumps(
            {"status": "error", "message": f"ffmpeg failed: {stderr}"},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("Unexpected error transforming %s", filepath)
        return json.dumps(
            {"status": "error", "message": f"Unexpected error: {e}"},
            ensure_ascii=False,
        )


def _fmt_to_ext(fmt: str) -> str:
    """Map detected format to a file extension."""
    return {"flac": "flac", "ogg": "ogg", "mp3": "mp3", "wma": "wma"}.get(fmt, "bin")
