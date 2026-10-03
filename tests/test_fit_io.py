"""FIT loading: fast numpy decoder vs the fitparse reference, CRC safety and fallbacks."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fitparse.utils import FitCRCError, FitEOFError, FitParseError

from cycling_tools import fitfast
from cycling_tools.fit_io import _messages, load_fit, FitFile

ROOT = Path(__file__).resolve().parent.parent
T0 = 1_000_000_000  # FIT seconds since 1989-12-31 (well above the 0x10000000 "absolute time" threshold)


def _defn(local: int, mesg: int, fields, dev: bool = False) -> bytes:
    h = 0x40 | local | (0x20 if dev else 0)
    b = bytes([h, 0, 0]) + struct.pack("<H", mesg) + bytes([len(fields)])
    for num, size, bt in fields:
        b += bytes([num, size, bt])
    if dev:
        b += bytes([0])
    return b


def make_fit(n: int = 120, *, compressed: bool = False, dev: bool = False, header_crc: bool = True,
             fix_crc: bool = True) -> bytes:
    """Tiny valid FIT file: n 1 Hz records (timestamp, power, speed, altitude) then one lap."""
    rec_fields = [(253, 4, 0x86), (7, 2, 0x84), (6, 2, 0x84), (2, 2, 0x84)]
    lap_fields = [(253, 4, 0x86), (2, 4, 0x86), (7, 4, 0x86), (9, 4, 0x86), (19, 2, 0x84)]
    body = _defn(0, 20, rec_fields, dev=dev)
    for i in range(n):
        if compressed and i == 5:  # compressed-timestamp data message (header bit 7 set, local type 0)
            body += bytes([0x80 | 3]) + struct.pack("<IHHH", 0, 200 + i, 5000 + 10 * i, 3000)
            continue
        body += bytes([0x00]) + struct.pack("<IHHH", T0 + i, 200 + i, 5000 + 10 * i, 3000)
    body += _defn(1, 19, lap_fields)
    body += bytes([0x01]) + struct.pack("<IIIIH", T0 + n, T0, n * 1000, n * 500, 215)
    hdr = bytes([14, 0x20]) + struct.pack("<HI", 2132, len(body)) + b".FIT"
    hdr += struct.pack("<H", fitfast._crc16_py(hdr) if header_crc else 0)
    data = hdr + body
    crc = fitfast._crc16_py(data) if fix_crc else 0xBEEF
    return data + struct.pack("<H", crc)


def _ref(data: bytes):
    import io
    fit = FitFile(io.BytesIO(data))
    return _messages(fit, "record"), _messages(fit, "lap")


def test_crc_python_and_numba_agree():
    rng = np.random.default_rng(1)
    b = rng.integers(0, 256, 5000, dtype=np.uint8).tobytes()
    assert fitfast.crc16(b) == fitfast._crc16_py(b)
    # file CRC of a valid file, computed over header+body, equals the stored trailer
    d = make_fit()
    assert fitfast.crc16(d[:-2]) == struct.unpack("<H", d[-2:])[0]


def test_fast_decoder_matches_fitparse_on_synthetic_file():
    d = make_fit()
    rec, lap = fitfast.read_messages(d)
    rrec, rlap = _ref(d)
    assert len(rec) == len(rrec) == 120
    for col in ("power", "speed", "altitude"):
        np.testing.assert_allclose(rec[col].to_numpy(), pd.to_numeric(rrec[col]).to_numpy())
    assert (pd.to_datetime(rec["timestamp"]) == pd.to_datetime(rrec["timestamp"])).all()
    assert lap["avg_power"].iloc[0] == rlap["avg_power"].iloc[0] == 215
    assert lap["start_time"].iloc[0] == pd.Timestamp(rlap["start_time"].iloc[0])


def test_load_fit_synthetic_and_laps_dtype():
    r = load_fit(make_fit(), name="x.fit")
    assert len(r.df) == 120
    assert r.laps["avg_power"].dtype == np.float64  # documented: always float64 (NaN-capable), fast path or fallback
    assert r.laps["avg_power"].iloc[0] == 215


@pytest.mark.parametrize("kwargs", [dict(compressed=True), dict(dev=True)])
def test_unsupported_features_fall_back(kwargs):
    d = make_fit(**kwargs)
    with pytest.raises(fitfast.FitDecodeError):
        fitfast.read_messages(d)
    # load_fit must hand the file to fitparse (the reference), never decode it with the fast path
    try:
        load_fit(d, name="c.fit")
    except Exception as exc:  # noqa: BLE001 - the hand-built file may be rejected by fitparse too; that is fine
        assert not isinstance(exc, fitfast.FitDecodeError)


def test_corrupt_data_byte_is_rejected_not_silently_accepted():
    d = bytearray(make_fit())
    d[14 + 18 + 1 + 5] ^= 0x10  # flip a bit inside a record's power field: framing stays valid, only the CRC catches it
    with pytest.raises(fitfast.FitDecodeError, match="CRC"):
        fitfast.read_messages(bytes(d))
    with pytest.raises(FitCRCError):  # ... and load_fit surfaces fitparse's error instead of returning bad data
        load_fit(bytes(d), name="bad.fit")


def test_bad_file_crc_and_bad_header_crc():
    with pytest.raises(FitCRCError):
        load_fit(make_fit(fix_crc=False), name="bad.fit")
    d = bytearray(make_fit())
    d[12] ^= 0x01  # header CRC mismatch
    with pytest.raises(fitfast.FitDecodeError, match="header CRC"):
        fitfast.read_messages(bytes(d))
    assert len(load_fit(make_fit(header_crc=False), name="h0.fit").df) == 120  # zero header CRC = not set: accepted


@pytest.mark.parametrize("cut", [5, 20, 200, -1, -2, -30])
def test_truncated_file_raises(cut):
    d = make_fit()
    d = d[:cut] if cut > 0 else d[:cut]
    with pytest.raises(fitfast.FitDecodeError):
        fitfast.read_messages(d)
    with pytest.raises((FitEOFError, FitParseError, ValueError, Exception)):
        load_fit(d, name="trunc.fit")


def test_trailing_garbage_or_chained_file_falls_back():
    d = make_fit()
    with pytest.raises(fitfast.FitDecodeError):
        fitfast.read_messages(d + make_fit())


@pytest.mark.parametrize("name", ["Manchester_District_TTA_50_WU_CD.fit", "southport.FIT"])
def test_real_files_match_fitparse(name):
    p = ROOT / "Fit_files" / name
    if not p.exists():
        pytest.skip("sample FIT file not present")
    data = p.read_bytes()
    rec, lap = fitfast.read_messages(data)
    rrec, rlap = _ref(data)
    assert len(rec) == len(rrec)
    for col in ("power", "heart_rate", "cadence"):
        if col in rrec:
            np.testing.assert_allclose(rec[col].to_numpy(), pd.to_numeric(rrec[col], errors="coerce").to_numpy(),
                                       equal_nan=True)
    assert len(lap) == len(rlap)
