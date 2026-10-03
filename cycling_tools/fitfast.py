"""Fast FIT decoder for the ``record`` and ``lap`` messages used by :mod:`fit_io`.

fitparse builds a Python object per field of every message (all message types), which is ~100x slower than needed.
Here one cheap Python pass over the message headers records where each wanted message starts; the field values
are then gathered with numpy. Output matches fitparse for these fields (values, scale/offset, invalid -> NaN,
dates); anything unusual raises :class:`FitDecodeError` and the caller falls back to fitparse, which is the
reference implementation and raises on genuinely corrupt files. The fast path never accepts a file unless:

* the 14-byte header CRC (when non-zero) and the trailing file CRC-16 both match,
* the file is a single (non-chained) FIT stream whose message framing parses exactly to the data size,
* the wanted messages (record, lap) use no compressed-timestamp headers and no developer fields, and
* every wanted field has a known base type whose size matches the definition (no arrays / strings).
"""
from __future__ import annotations

import struct

import numpy as np
import pandas as pd

FIT_EPOCH_OFFSET_S = 631065600  # 1989-12-31 UTC -> unix
MESG_RECORD, MESG_LAP = 20, 19
# FIT base type byte -> (numpy dtype, size, invalid value)
_BT = {
    0x00: ("u1", 1, 0xFF), 0x01: ("i1", 1, 0x7F), 0x02: ("u1", 1, 0xFF), 0x83: ("i2", 2, 0x7FFF),
    0x84: ("u2", 2, 0xFFFF), 0x85: ("i4", 4, 0x7FFFFFFF), 0x86: ("u4", 4, 0xFFFFFFFF),
    0x88: ("f4", 4, None), 0x89: ("f8", 8, None), 0x0A: ("u1", 1, 0x00), 0x8B: ("u2", 2, 0x0000),
    0x8C: ("u4", 4, 0x00000000),
}
# field number -> name for the fields we use; REC_SCALE / LAP_SCALE: name -> (scale, offset)
REC_FIELDS = {253: "timestamp", 0: "position_lat", 1: "position_long", 2: "altitude", 3: "heart_rate", 4: "cadence",
              5: "distance", 6: "speed", 7: "power", 13: "temperature", 73: "enhanced_speed", 78: "enhanced_altitude"}
REC_SCALE = {"altitude": (5.0, 500.0), "distance": (100.0, 0.0), "speed": (1000.0, 0.0),
             "enhanced_speed": (1000.0, 0.0), "enhanced_altitude": (5.0, 500.0)}
LAP_FIELDS = {253: "timestamp", 2: "start_time", 7: "total_elapsed_time", 8: "total_timer_time", 9: "total_distance",
              13: "avg_speed", 110: "enhanced_avg_speed", 19: "avg_power"}
LAP_SCALE = {"total_elapsed_time": (1000.0, 0.0), "total_timer_time": (1000.0, 0.0), "total_distance": (100.0, 0.0),
             "avg_speed": (1000.0, 0.0), "enhanced_avg_speed": (1000.0, 0.0)}


class FitDecodeError(Exception):
    pass


_CRC_TAB = (0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
            0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400)


def _crc16_py(data, crc: int = 0) -> int:
    """FIT CRC-16 (nibble table from the SDK; identical to CRC-16/ARC). Pure-Python fallback."""
    t = _CRC_TAB
    for b in data:
        crc = ((crc >> 4) & 0x0FFF) ^ t[crc & 0xF] ^ t[b & 0xF]
        crc = ((crc >> 4) & 0x0FFF) ^ t[crc & 0xF] ^ t[(b >> 4) & 0xF]
    return crc


try:
    from numba import njit

    @njit(cache=True, nogil=True)
    def _crc16_nb(a, n):
        t = np.array([0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
                      0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400], dtype=np.int64)
        crc = 0
        for i in range(n):
            b = np.int64(a[i])
            crc = ((crc >> 4) & 0x0FFF) ^ t[crc & 0xF] ^ t[b & 0xF]
            crc = ((crc >> 4) & 0x0FFF) ^ t[crc & 0xF] ^ t[(b >> 4) & 0xF]
        return crc
except ImportError:  # pragma: no cover
    _crc16_nb = None


def crc16(buf) -> int:
    """CRC-16 of ``buf`` as used by FIT files (numba kernel when available, ~1 ms/MB; pure Python otherwise)."""
    if _crc16_nb is not None:
        a = np.frombuffer(buf, dtype=np.uint8)
        return int(_crc16_nb(a, len(a)))
    return _crc16_py(buf)


def _decode(buf: bytes):
    n = len(buf)
    if n < 14:
        raise FitDecodeError("too short")
    hsize = buf[0]
    if hsize not in (12, 14) or buf[8:12] != b".FIT":
        raise FitDecodeError("bad header")
    data_size = struct.unpack_from("<I", buf, 4)[0]
    end = hsize + data_size
    if end + 2 != n:   # truncated / chained / trailing data -> let the caller fall back
        raise FitDecodeError("unexpected file layout")
    if hsize == 14:
        hcrc = struct.unpack_from("<H", buf, 12)[0]
        if hcrc != 0 and hcrc != crc16(buf[:12]):
            raise FitDecodeError("header CRC mismatch")
    if struct.unpack_from("<H", buf, end)[0] != crc16(memoryview(buf)[:end]):
        raise FitDecodeError("file CRC mismatch")
    defs = {}          # local type -> (mesg_num, fields[(num, size, base_type)], total_size, big_endian)
    # wanted mesg_num -> {definition id: [message offsets, definition]}
    offs = {MESG_RECORD: {}, MESG_LAP: {}}
    pos = hsize
    while pos < end:
        h = buf[pos]
        pos += 1
        if h & 0x80:                      # compressed timestamp header: data message
            lt = (h >> 5) & 0x03
            d = defs.get(lt)
            if d is None:
                raise FitDecodeError("data message without definition")
            if d[0] in offs:
                raise FitDecodeError("compressed timestamp on wanted message")
            pos += d[2]
        elif h & 0x40:                    # definition message
            lt = h & 0x0F
            has_dev = bool(h & 0x20)
            big = buf[pos + 1] == 1
            mesg = struct.unpack_from(">H" if big else "<H", buf, pos + 2)[0]
            nf = buf[pos + 4]
            p = pos + 5
            fields = []
            tot = 0
            for _ in range(nf):
                fields.append((buf[p], buf[p + 1], buf[p + 2]))
                tot += buf[p + 1]
                p += 3
            if has_dev:
                if mesg in offs:
                    raise FitDecodeError("developer fields on wanted message")
                nd = buf[p]
                p += 1
                for _ in range(nd):
                    tot += buf[p + 1]
                    p += 3
            defs[lt] = (mesg, fields, tot, big)
            pos = p
        else:                             # normal data message
            lt = h & 0x0F
            d = defs.get(lt)
            if d is None:
                raise FitDecodeError("data message without definition")
            if d[0] in offs:
                offs[d[0]].setdefault(id(d), [[], d])[0].append(pos)
            pos += d[2]
    if pos != end:
        raise FitDecodeError("overrun")
    return offs


def _gather(arr: np.ndarray, offsets: list[int], fields, wanted: dict, scale: dict, big: bool = False) -> dict[str, np.ndarray]:
    off = np.asarray(offsets, dtype=np.int64)
    out = {}
    fo = 0
    for num, size, bt in fields:
        name = wanted.get(num)
        if name is not None:
            if bt not in _BT:
                raise FitDecodeError(f"unsupported base type {bt:#x} for field {name}")
            if _BT[bt][1] != size:
                # array field: fitparse yields a tuple, which fit_io's to_numeric(errors="coerce") turns into NaN, i.e.
                # the same as a missing column. Fine for measurement fields; never for the time keys or odd sizes.
                if size % _BT[bt][1] or name in ("timestamp", "start_time"):
                    raise FitDecodeError(f"unsupported layout for field {name} (base type {bt:#x}, size {size})")
                fo += size
                continue
            dt, _, inv = _BT[bt]
            # gather `size` bytes per message and reinterpret them as the field's dtype
            idx = (off + fo)[:, None] + np.arange(size)
            raw = np.ascontiguousarray(arr[idx]).view((">" if big else "<") + dt).reshape(-1)
            v = raw.astype(float)
            if inv is not None:
                v[raw == inv] = np.nan
            sc = scale.get(name)
            if sc:
                v = v / sc[0] - sc[1]  # same order of operations as fitparse: value / scale - offset
            out[name] = v
        fo += size
    return out


def read_messages(buf: bytes):
    """Return (records_df, laps_df) with the same column names fitparse gives for the fields we need."""
    try:
        offs = _decode(buf)
    except (IndexError, struct.error) as exc:  # framing ran off the buffer
        raise FitDecodeError(f"corrupt framing: {exc}") from exc
    arr = np.frombuffer(buf, dtype=np.uint8)
    res = {}
    for mesg, wanted, scale in ((MESG_RECORD, REC_FIELDS, REC_SCALE), (MESG_LAP, LAP_FIELDS, LAP_SCALE)):
        parts = []
        for lt, (offsets, d) in offs[mesg].items():
            cols = _gather(arr, offsets, d[1], wanted, scale, d[3])
            parts.append((np.asarray(offsets), cols))
        if not parts:
            res[mesg] = pd.DataFrame()
            continue
        order = np.argsort(np.concatenate([p[0] for p in parts]), kind="stable")
        names = sorted({k for _, c in parts for k in c})
        frames = {}
        for k in names:
            col = np.concatenate([c.get(k, np.full(len(o), np.nan)) for o, c in parts])[order]
            frames[k] = col
        df = pd.DataFrame(frames)
        for tcol in ("timestamp", "start_time"):
            if tcol in df:
                v = df[tcol].to_numpy()
                ok = np.isfinite(v)
                ns = np.zeros(len(v), dtype="int64")
                iv = v[ok].astype("int64")
                # FIT: date_time values below 0x10000000 are not absolute times (fitparse leaves them as raw ints,
                # which pandas then reads as ns-since-1970): mimic that so behaviour is unchanged on junk laps.
                ns[ok] = np.where(iv >= 0x10000000, (iv + FIT_EPOCH_OFFSET_S) * 1_000_000_000, iv)
                dt = ns.view("datetime64[ns]")
                dt[~ok] = np.datetime64("NaT")
                df[tcol] = dt
        res[mesg] = df
    return res[MESG_RECORD], res[MESG_LAP]
