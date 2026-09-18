from pathlib import Path

import pandas as pd

INPUT_DIR = Path(__file__).resolve().parent / ".." / "data" / "reference_values"
REFERENCES_PATH = INPUT_DIR / "references.csv"
OUTPUT_PATH = INPUT_DIR / "median_1s_chunk.csv"

FRAME_STEP_MS = 2  # 1 time_index = 2 ms
CHUNK_MS = 1000
CHUNK_INDICES = CHUNK_MS // FRAME_STEP_MS  # 500 frames in the first 1 s

PARAM_NAMES = [
    "loudness_zwtv",
    "sharpness_din_tv",
    "roughness_dw",
    "tnr_ecma_perseg",
    "sii_ansi",
]


def build_median_1s_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Median of each parameter per time_index over the first 1000 ms,
    pooled across all source files.

    Each parameter keeps its native resolution on the shared 2 ms grid
    (loudness/sharpness every 2 ms, roughness every 100 ms, tnr every 500 ms,
    sii once per second); time indices without a value stay NaN, so the
    values remain aligned to their correct time_index.
    """
    chunk = df[df["time_index"] < CHUNK_INDICES]
    return chunk.groupby("time_index", observed=True)[PARAM_NAMES].median().reset_index()


def main():
    print(f"Reading {REFERENCES_PATH} ...")
    df = pd.read_csv(
        REFERENCES_PATH,
        usecols=["time_index"] + PARAM_NAMES,
        dtype={"time_index": "int32"} | {p: "float32" for p in PARAM_NAMES},
    )
    print(f"Loaded {len(df):,} rows")

    out = build_median_1s_chunk(df)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH} ({len(out)} time indices)")


if __name__ == "__main__":
    main()