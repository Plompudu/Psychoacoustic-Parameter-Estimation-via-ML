PARAM_NAMES = [
    "loudness_zwtv",
    "sharpness_din_tv",
    "roughness_dw",
    "tnr_ecma_perseg",
    "sii_ansi",
]

# Frame counts are deterministic per parameter for 1 s @ 48 kHz audio.
FRAME_COUNTS: dict[str, int] = {
    "loudness_zwtv": 500,
    "sharpness_din_tv": 500,
    "roughness_dw": 9,
    "tnr_ecma_perseg": 2,
    "sii_ansi": 1,
}
