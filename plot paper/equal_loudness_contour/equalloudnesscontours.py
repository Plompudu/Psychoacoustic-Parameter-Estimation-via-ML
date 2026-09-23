
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal

# ── style ───────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}
plt.rcParams.update(STYLE)

_cool = plt.cm.cool(np.linspace(0, 1, 9))

# ── Figure 1: Equal-loudness contours (ISO 226:2003) ─────────────────────────

def iso226(phon, freqs):
    """
    Returns SPL values for a given loudness level (phon) at each frequency,
    per ISO 226:2003 Table 1 coefficients.
    """
    # Reference frequencies and coefficients from ISO 226:2003, Table 1
    f_ref = np.array([
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500
    ], dtype=float)

    af = np.array([
        0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330,
        0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244,
        0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301
    ])

    Lu = np.array([
        -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5,
        -3.1,  -2.0,  -1.1,  -0.4,   0.0,   0.3,   0.5,   0.0, -2.7, -4.1,
        -1.0,   1.7,   2.5,   1.2,  -2.1,  -7.1, -11.2, -10.7,  -3.1
    ])

    Tf = np.array([
        78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
        14.4, 11.4,  8.6,  6.2,  4.4,  3.0,  2.2,  2.4,  3.5,  1.7,
        -1.3, -4.2, -6.0, -5.4, -1.5,  6.0, 12.6, 13.9, 12.3
    ])

    # Interpolate coefficients to requested frequencies
    af_i  = np.interp(freqs, f_ref, af)
    Lu_i  = np.interp(freqs, f_ref, Lu)
    Tf_i  = np.interp(freqs, f_ref, Tf)

    Ln = phon
    Af = 0.00447 * (10 ** (0.025 * Ln) - 1.15) + (0.4 * 10 ** ((Tf_i + Lu_i) / 10 - 9)) ** af_i
    Lp = (10 / af_i) * np.log10(Af) - Lu_i + 94
    return Lp


freqs = np.logspace(np.log10(20), np.log10(12500), 500)
phon_levels = [10, 20, 40, 60, 80, 100]
colors = plt.cm.cool(np.linspace(0.15, 1.0, len(phon_levels)))

fig1, ax1 = plt.subplots(figsize=(7, 4.5))

for phon, color in zip(phon_levels, colors):
    spl = iso226(phon, freqs)
    ax1.semilogx(freqs, spl, color=color, linewidth=2.5, label=f"{phon} phon")

ax1.set_xlim(20, 12500)
ax1.set_ylim(0, 120)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Sound Pressure Level (dB SPL)")
# ax1.set_title("Equal-Loudness-Level Contours (ISO 226:2003)")
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"
))
ax1.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 5]))
ax1.legend(title="Loudness level", loc="upper right", framealpha=0.9)
ax1.annotate("ISO 226:2003", xy=(0.02, 0.04), xycoords="axes fraction",
             fontsize=8, color="#888888")

fig1.tight_layout()
output_dir = "."
fig1.savefig(f"{output_dir}/equal_loudness_contours.pdf", dpi=300, bbox_inches="tight")
fig1.savefig(f"{output_dir}/equal_loudness_contours.png", dpi=300, bbox_inches="tight")
