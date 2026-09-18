from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .params import PARAM_NAMES

SEGMENT_PATTERN = re.compile(r"^\[(?P<idx>\d+)\]_(?P<chan>mono|left|right)_(?P<parent>.+)$")

WINDOW_FRAMES = 500  # frames per 1 s segment at the 2 ms time grid


def parse_segment_name(stem: str) -> tuple[int, str, str] | None:
    """Return (segment_index, channel, parent_source) for a WAV stem like ``[00003]_mono_[000001]_x_ch1``."""
    match = SEGMENT_PATTERN.match(stem)
    if match is None:
        return None
    return int(match.group("idx")), match.group("chan"), match.group("parent")


def collect_segment_stems(directory: str | Path) -> list[str]:
    """Sorted unique WAV stems in ``directory`` (e.g. data/raw_sound_files_1s)."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    return sorted({p.stem for p in dir_path.glob("*.wav")})


def segment_local_offsets(stems: list[str]) -> dict[tuple[int, str], int]:
    """Map (segment_index, parent) -> local chunk position within that parent.

    ``1_split_audio_into_1s_segments.py`` numbers segments with a running
    counter that continues across source files, so ``[00187]_mono_<parent>``
    is not necessarily near the start of ``parent``. Within one parent the
    counter forms a contiguous block, so the local chunk position of a stem
    is simply its rank among that parent's segment indices. The ``parent``
    reference table is indexed by local position (window = local * 500 frames).
    """
    by_parent: dict[str, list[int]] = {}
    for stem in stems:
        parsed = parse_segment_name(stem)
        if parsed is None:
            continue
        idx, _channel, parent = parsed
        by_parent.setdefault(parent, []).append(idx)
    offsets: dict[tuple[int, str], int] = {}
    for parent, indices in by_parent.items():
        for local, idx in enumerate(sorted(set(indices))):
            offsets[(idx, parent)] = local
    return offsets


def load_references_grouped(
    path: str | Path,
    columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read data/reference_values/references.csv and group rows by source_file."""
    columns = columns or (["source_file", "time_index"] + PARAM_NAMES)
    dtype: dict[str, str] = {"time_index": "int32"}
    dtype.update({name: "float32" for name in PARAM_NAMES})
    df = pd.read_csv(path, usecols=columns, dtype=dtype)
    return {parent: group for parent, group in df.groupby("source_file", observed=True, sort=False)}


def build_labels_for_stem(
    ref: pd.DataFrame,
    seg_idx: int,
    window: int = WINDOW_FRAMES,
) -> dict[str, np.ndarray] | None:
    """Slice one 1 s segment window out of a parent's reference table.

    Returns ``{param: np.ndarray[n] float32}`` where ``n`` is the number of
    frames covered by the window (``<= window``; shorter only for a final
    partial segment). Params without a sample in a given frame hold NaN.
    """
    time_index = ref["time_index"].to_numpy()
    low = seg_idx * window
    high = low + window
    mask = (time_index >= low) & (time_index < high)
    rel = time_index[mask] - low
    if rel.size == 0:
        return None
    n = int(rel.max()) + 1
    labels: dict[str, np.ndarray] = {}
    for name in PARAM_NAMES:
        values = np.empty(n, dtype=np.float32)
        values[:] = np.nan
        values[rel] = ref[name].to_numpy(dtype=np.float32)[mask]
        labels[name] = values
    return labels


def build_labels_for_stems(
    refs: dict[str, pd.DataFrame],
    stems: list[str],
    offsets: dict[tuple[int, str], int] | None = None,
    window: int = WINDOW_FRAMES,
) -> dict[str, dict[str, np.ndarray]]:
    """Build per-segment labels for every parseable stem present in ``refs``.

    ``offsets`` maps each stem's (segment_index, parent) to its local chunk
    position within the parent (see ``segment_local_offsets``). When omitted,
    the segment_index is treated as the local chunk position directly.
    """
    labels: dict[str, dict[str, np.ndarray]] = {}
    for stem in stems:
        parsed = parse_segment_name(stem)
        if parsed is None:
            continue
        seg_idx, _channel, parent = parsed
        ref = refs.get(parent)
        if ref is None:
            continue
        if offsets is not None:
            local = offsets.get((seg_idx, parent))
            if local is None:
                continue
            seg_idx = local
        segment_labels = build_labels_for_stem(ref, seg_idx, window=window)
        if segment_labels is not None:
            labels[stem] = segment_labels
    return labels