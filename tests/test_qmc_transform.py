"""Tests for QQ Music decryption and M4A transcoding."""

import json
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from personal_agent.tools.qmc_transform import (
    _QMC_V1_KEY,
    _QMC_MAP_SEED,
    _xor_bytes,
    _decrypt_qmc_v1,
    _decrypt_map,
    _fmt_to_ext,
    decrypt_qmc,
    detect_format,
    transcode_to_m4a,
    transform_qmc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_encrypted_v1(plaintext: bytes) -> bytes:
    """Encrypt data with the QMC v1 key (round-trip helper)."""
    return _xor_bytes(plaintext, _QMC_V1_KEY)


def _make_map_file(plaintext: bytes) -> bytes:
    """Build a valid .mflac file with an embedded map-cipher key.

    Encrypts *plaintext* with a known key, then embeds that key (XOR'd
    with the map seed) in the footer so ``_decrypt_map`` can recover it.
    """
    # Use a fixed key so the test is deterministic.
    real_key = bytes(range(256))
    encrypted_audio = _xor_bytes(plaintext, real_key)
    encrypted_key_data = _xor_bytes(real_key, _QMC_MAP_SEED)
    return encrypted_audio + encrypted_key_data + struct.pack("<I", len(encrypted_key_data))


# ---------------------------------------------------------------------------
# _xor_bytes
# ---------------------------------------------------------------------------


def test_xor_bytes_with_repeating_key():
    result = _xor_bytes(b"\x01\x02\x03\x04", b"\x01")
    assert result == b"\x00\x03\x02\x05"


def test_xor_bytes_roundtrip():
    data = b"hello world test data"
    key = b"\x5a\xa5"
    encrypted = _xor_bytes(data, key)
    assert _xor_bytes(encrypted, key) == data


def test_xor_bytes_empty():
    assert _xor_bytes(b"", b"\xff") == b""


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


def test_detect_flac():
    assert detect_format(b"fLaC\x00\x00\x00") == "flac"


def test_detect_ogg():
    assert detect_format(b"OggS\x00\x02\x00") == "ogg"


def test_detect_mp3_sync_header():
    assert detect_format(b"\xff\xfb\x90\x00") == "mp3"


def test_detect_mp3_alt():
    assert detect_format(b"\xff\xf3\x00\x00") == "mp3"


def test_detect_wma():
    data = bytearray(10)
    data[4:8] = b"\x30\x26\xb2\x75"
    assert detect_format(bytes(data)) == "wma"


def test_detect_unknown():
    assert detect_format(b"\x00\x00\x00\x00\x00") == "unknown"


# ---------------------------------------------------------------------------
# _decrypt_qmc_v1
# ---------------------------------------------------------------------------


def test_decrypt_qmc_v1_roundtrip():
    plaintext = b"fLaC" + bytes(1000)  # fake FLAC header
    encrypted = _make_encrypted_v1(plaintext)
    result = _decrypt_qmc_v1(encrypted)
    assert result == plaintext


def test_decrypt_qmc_v1_empty():
    assert _decrypt_qmc_v1(b"") == b""


# ---------------------------------------------------------------------------
# _decrypt_map
# ---------------------------------------------------------------------------


def test_decrypt_map_roundtrip():
    plaintext = b"fLaC" + bytes(500)
    file_data = _make_map_file(plaintext)
    result = _decrypt_map(file_data)
    assert result == plaintext


def test_decrypt_map_too_small():
    with pytest.raises(ValueError, match="too small"):
        _decrypt_map(b"123456789")  # < 10 bytes


def test_decrypt_map_invalid_key_length():
    # last 4 bytes claim key is huge
    data = bytes(100) + struct.pack("<I", 200)  # key bigger than file
    with pytest.raises(ValueError, match="invalid key length"):
        _decrypt_map(data)


# ---------------------------------------------------------------------------
# decrypt_qmc
# ---------------------------------------------------------------------------


def test_decrypt_qmc_v1_file(temp_dir):
    plaintext = b"fLaC" + bytes(500)
    encrypted = _make_encrypted_v1(plaintext)
    path = temp_dir / "song.qmcflac"
    path.write_bytes(encrypted)

    result = decrypt_qmc(path)
    assert result == plaintext


def test_decrypt_qmc_mflac_file(temp_dir):
    plaintext = b"fLaC" + bytes(500)
    file_data = _make_map_file(plaintext)
    path = temp_dir / "song.mflac"
    path.write_bytes(file_data)

    result = decrypt_qmc(path)
    assert result == plaintext


def test_decrypt_qmc_mgg_file(temp_dir):
    plaintext = b"OggS" + bytes(500)
    file_data = _make_map_file(plaintext)
    path = temp_dir / "song.mgg"
    path.write_bytes(file_data)

    result = decrypt_qmc(path)
    assert result == plaintext


def test_decrypt_qmc_file_not_found(temp_dir):
    with pytest.raises(FileNotFoundError):
        decrypt_qmc(temp_dir / "nonexistent.qmcflac")


def test_decrypt_qmc_not_a_file(temp_dir):
    with pytest.raises(ValueError, match="Not a file"):
        decrypt_qmc(temp_dir)


def test_decrypt_qmc_unsupported_extension(temp_dir):
    path = temp_dir / "song.txt"
    path.write_text("not music")
    with pytest.raises(ValueError, match="Unsupported format"):
        decrypt_qmc(path)


def test_decrypt_qmc_empty_file(temp_dir):
    path = temp_dir / "empty.qmcflac"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        decrypt_qmc(path)


# ---------------------------------------------------------------------------
# transcode_to_m4a
# ---------------------------------------------------------------------------


def test_transcode_flac_to_m4a():
    with patch("subprocess.run") as mock_run, patch(
        "shutil.which", return_value="/usr/local/bin/ffmpeg"
    ):
        transcode_to_m4a(
            Path("/tmp/test.flac"),
            Path("/tmp/test.m4a"),
            audio_format="flac",
        )
        cmd = mock_run.call_args[0][0]
        assert "-acodec" in cmd
        assert "alac" in cmd
        assert str(Path("/tmp/test.m4a")) in cmd


def test_transcode_mp3_to_m4a():
    with patch("subprocess.run") as mock_run, patch(
        "shutil.which", return_value="/usr/local/bin/ffmpeg"
    ):
        transcode_to_m4a(
            Path("/tmp/test.mp3"),
            Path("/tmp/test.m4a"),
            audio_format="mp3",
        )
        cmd = mock_run.call_args[0][0]
        assert "aac" in cmd
        assert "256k" in cmd


def test_transcode_ffmpeg_not_found():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            transcode_to_m4a(Path("/tmp/a.flac"), Path("/tmp/a.m4a"))


def test_transcode_ffmpeg_failure():
    with patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"codec error"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            transcode_to_m4a(Path("/tmp/a.flac"), Path("/tmp/a.m4a"))


# ---------------------------------------------------------------------------
# transform_qmc (agent tool)
# ---------------------------------------------------------------------------


def test_transform_qmc_success(temp_dir):
    plaintext = b"fLaC" + bytes(500)
    encrypted = _make_encrypted_v1(plaintext)
    input_path = temp_dir / "song.qmcflac"
    input_path.write_bytes(encrypted)

    with patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), patch(
        "subprocess.run"
    ) as mock_run:
        result = json.loads(transform_qmc(str(input_path)))
        assert result["status"] == "success"
        assert result["format"] == "flac"
        assert Path(result["original"]) == input_path.resolve()
        mock_run.assert_called_once()
        assert Path(result["output_path"]).suffix == ".m4a"


def test_transform_qmc_file_not_found():
    result = json.loads(transform_qmc("/nonexistent/song.qmcflac"))
    assert result["status"] == "error"


def test_transform_qmc_unsupported_extension(temp_dir):
    path = temp_dir / "song.txt"
    path.write_text("hello")
    result = json.loads(transform_qmc(str(path)))
    assert result["status"] == "error"
    assert "Unsupported" in result["message"]


def test_transform_qmc_custom_output_dir(temp_dir):
    plaintext = b"fLaC" + bytes(500)
    encrypted = _make_encrypted_v1(plaintext)
    input_path = temp_dir / "song.qmcflac"
    input_path.write_bytes(encrypted)

    out_dir = temp_dir / "converted"
    out_dir.mkdir()

    with patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), patch(
        "subprocess.run"
    ):
        result = json.loads(transform_qmc(str(input_path), output_dir=str(out_dir)))
        assert result["status"] == "success"
        assert str(out_dir) in result["output_path"]


def test_transform_qmc_ffmpeg_missing(temp_dir):
    plaintext = b"fLaC" + bytes(500)
    encrypted = _make_encrypted_v1(plaintext)
    input_path = temp_dir / "song.qmcflac"
    input_path.write_bytes(encrypted)

    with patch("shutil.which", return_value=None):
        result = json.loads(transform_qmc(str(input_path)))
        assert result["status"] == "error"
        assert "ffmpeg" in result["message"]


def test_transform_plain_ogg_to_m4a(temp_dir):
    input_path = temp_dir / "song.ogg"
    input_path.write_bytes(b"OggS" + bytes(500))

    with patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), patch(
        "subprocess.run"
    ) as mock_run:
        result = json.loads(transform_qmc(str(input_path)))
        assert result["status"] == "success"
        assert result["format"] == "ogg"
        cmd = mock_run.call_args[0][0]
        assert "aac" in cmd


def test_transform_plain_flac_to_m4a(temp_dir):
    input_path = temp_dir / "song.flac"
    input_path.write_bytes(b"fLaC" + bytes(500))

    with patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), patch(
        "subprocess.run"
    ) as mock_run:
        result = json.loads(transform_qmc(str(input_path)))
        assert result["status"] == "success"
        assert result["format"] == "flac"
        cmd = mock_run.call_args[0][0]
        assert "alac" in cmd


def test_transform_plain_mp3_to_m4a(temp_dir):
    input_path = temp_dir / "song.mp3"
    input_path.write_bytes(b"\xff\xfb" + bytes(500))

    with patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), patch(
        "subprocess.run"
    ) as mock_run:
        result = json.loads(transform_qmc(str(input_path)))
        assert result["status"] == "success"
        assert result["format"] == "mp3"
        cmd = mock_run.call_args[0][0]
        assert "aac" in cmd


def test_transform_plain_unsupported_extension(temp_dir):
    path = temp_dir / "song.txt"
    path.write_text("not audio")
    result = json.loads(transform_qmc(str(path)))
    assert result["status"] == "error"
    assert "Unsupported" in result["message"]


# ---------------------------------------------------------------------------
# _fmt_to_ext
# ---------------------------------------------------------------------------


def test_fmt_to_ext_known():
    assert _fmt_to_ext("flac") == "flac"
    assert _fmt_to_ext("mp3") == "mp3"
    assert _fmt_to_ext("ogg") == "ogg"


def test_fmt_to_ext_unknown():
    assert _fmt_to_ext("unknown") == "bin"
