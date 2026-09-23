import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal

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
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}
plt.rcParams.update(STYLE)

fs = 48000

b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
a1 = np.array([1.0,              -1.69065929318241, 0.73248077421585])
b2 = np.array([1.0, -2.0, 1.0])
a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])

w, h1 = signal.freqz(b1, a1, worN=8192, fs=fs)
_, h2 = signal.freqz(b2, a2, worN=8192, fs=fs)
h_total = h1 * h2

mask = (w >= 20) & (w <= 20000)

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.semilogx(w[mask], 20 * np.log10(np.abs(h_total[mask])),
            color=plt.cm.cool(0.5), linewidth=2.5)

ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
ax.set_xlim(20, 20000)
ax.set_ylim(-15, 5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Level (dB)")
# ax.set_title("K-Weighting Frequency Response (ITU-R BS.1770-5)")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"
))
ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 5]))
ax.annotate("ITU-R BS.1770-5", xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=8, color="#888888")

fig.tight_layout()
output_dir = "."
fig.savefig(f"{output_dir}/k_weighting.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{output_dir}/k_weighting.png", dpi=300, bbox_inches="tight")
