import numpy as np
from math import sin, cos, log10, sqrt, pi


def gain(signal, db_gain: float):
    """
    Simple gain in dB.
    """
    linear = 10 ** (db_gain / 20)
    return signal * linear




def _biquad_filter(x, b, a):
    """
    Direct Form I biquad filter.
    b = [b0, b1, b2], a = [1, a1, a2]
    """
    y = np.zeros_like(x, dtype=float)
    x1 = x2 = y1 = y2 = 0.0

    b0, b1, b2 = b
    a0, a1, a2 = a

    for i in range(len(x)):
        y[i] = (b0/a0)*x[i] + (b1/a0)*x1 + (b2/a0)*x2 \
               - (a1/a0)*y1 - (a2/a0)*y2

        x2, x1 = x1, x[i]
        y2, y1 = y1, y[i]

    return y


def peaking_eq(fs, f0, Q, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * pi * f0 / fs
    alpha = sin(w0) / (2 * Q)

    cosw0 = cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cosw0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw0
    a2 = 1 - alpha / A

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def high_shelf(fs, f0, S, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * pi * f0 / fs
    alpha = sin(w0) / 2 * sqrt((A + 1/A) * (1/S - 1) + 2)
    cosw0 = cos(w0)

    b0 = A * ((A+1) + (A-1)*cosw0 + 2*sqrt(A)*alpha)
    b1 = -2*A * ((A-1) + (A+1)*cosw0)
    b2 = A * ((A+1) + (A-1)*cosw0 - 2*sqrt(A)*alpha)
    a0 = (A+1) - (A-1)*cosw0 + 2*sqrt(A)*alpha
    a1 = 2 * ((A-1) - (A+1)*cosw0)
    a2 = (A+1) - (A-1)*cosw0 - 2*sqrt(A)*alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def low_shelf(fs, f0, S, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * pi * f0 / fs
    alpha = sin(w0) / 2 * sqrt((A + 1/A) * (1/S - 1) + 2)
    cosw0 = cos(w0)

    b0 = A * ((A+1) - (A-1)*cosw0 + 2*sqrt(A)*alpha)
    b1 = 2*A * ((A-1) - (A+1)*cosw0)
    b2 = A * ((A+1) - (A-1)*cosw0 - 2*sqrt(A)*alpha)
    a0 = (A+1) + (A-1)*cosw0 + 2*sqrt(A)*alpha
    a1 = -2 * ((A-1) + (A+1)*cosw0)
    a2 = (A+1) + (A-1)*cosw0 - 2*sqrt(A)*alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def three_band_parametric_eq(signal, fs, bands):
    """
    bands: list of dicts like:
    {
        "type": "bell" | "low_shelf" | "high_shelf",
        "f0": frequency,
        "gain_db": gain,
        "Q": width
    }
    """
    filters = {
        "bell": peaking_eq,
        "low_shelf": low_shelf,
        "high_shelf": high_shelf,
    }

    out = signal.copy()

    for band in bands:
        bq, aq = filters[band["type"]](
            fs,
            band["f0"],
            band.get("Q", 1.0),
            band["gain_db"],
        )
        out = _biquad_filter(out, bq, aq)

    return out


def compressor(signal, threshold_db=-20, ratio=4, attack=0.01, release=0.1, fs=48000):
    """
    Simple feed-forward compressor using RMS detection.
    """
    signal = signal.astype(float)

    threshold = 10 ** (threshold_db / 20)

    attack_coeff = np.exp(-1 / (fs * attack))
    release_coeff = np.exp(-1 / (fs * release))

    env = 0.0
    out = np.zeros_like(signal)

    for i, x in enumerate(signal):
        x_abs = abs(x)

        # envelope follower
        if x_abs > env:
            env = attack_coeff * (env - x_abs) + x_abs
        else:
            env = release_coeff * (env - x_abs) + x_abs

        # gain computation
        if env > threshold:
            gain_db = -((20 * np.log10(env / threshold)) * (1 - 1/ratio))
            gain_lin = 10 ** (gain_db / 20)
        else:
            gain_lin = 1.0

        out[i] = x * gain_lin

    return out

def generate_randomized_training_set_with_applied_filters(input_folder, output_folder, number_of_samples):


    #generate random parameter set
    DB_GAIN_RANGE = (-40, 40)
    THRESHOLD_DB_RANGE = (-20, 10)
    RATIO_RANGE = (1, 10)
    ATTACK_RANGE = (0.01, 0.1)
    RELEASE_RANGE = (0.1, 1)
    TYPE_RANGE = ["bell", "low_shelf", "high_shelf"]
    F0_RANGE = (20, 20000)
    GAIN_DB_RANGE = (-40, 40)
    Q_RANGE = (0.1, 10)

    # load files
    #apply random parameters per file
    filtered_signal = gain(signal=signal, db_gain=db_gain)
    filtered_signal = compressor(signal=filtered_signal, threshold_db=threshold_db, ratio=ratio, attack=attack, release=release, fs=48000)
    bands = [
        {"type": type_1, "f0": f0_1, "gain_db": gain_db_1, "Q": Q_1},
        {"type": type_2, "f0": f0_2, "gain_db": gain_db_2, "Q": Q_2},
        {"type": type_3, "f0": f0_3, "gain_db": gain_db_3, "Q": Q_3},
    ]
    filtered_signal = three_band_parametric_eq(signal=filtered_signal, fs=48000, bands=bands)
    # save filtered_signal at output_path
    return