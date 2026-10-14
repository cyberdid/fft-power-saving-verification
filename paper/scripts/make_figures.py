"""Publication figures for the article (v5: strict construction is the primary object).

Inputs (all in paper/data/):
  strict_tables.csv        strict-mode ASAP depth, layers, shears, scales (strict_tables.py)
  depth_model_strict.csv   strict depth model d(2^e) (depth_exact_rule.py --strict)
  dft_precision.csv        full-DFT and Toeplitz precision runs (mp_check_dft.py, mp_check_uniform.py)
Default-mode values (zero multipliers dropped, seed = r) are the v4 table values below; they are
drawn only as thin dashed reference curves.

Outputs in paper/figures/: fig1_depth, fig2_cost, fig3_log4_law, fig4_tradeoff_r512,
fig5_precision (.pdf vector, .png 300 dpi). Widths: one column 84 mm, two columns 174 mm.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# Figures read from DATA and write to FIG. Override with --data-dir / --fig-dir so a fresh
# run can build figures from its own collected CSVs instead of the archived paper/data/.
import sys as _sys  # noqa: E402
_a = _sys.argv
DATA = _a[_a.index("--data-dir") + 1] if "--data-dir" in _a else os.path.join(ROOT, "paper", "data")
FIG = _a[_a.index("--fig-dir") + 1] if "--fig-dir" in _a else os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito palette (colour-blind safe)
BLUE, ORANGE, GREEN, BLACK, VERM, SKY, PURPLE = (
    "#0072B2", "#E69F00", "#009E73", "#000000", "#D55E00", "#56B4E9", "#CC79A7")

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
W1, W2 = 84 / 25.4, 174 / 25.4


def sci(x, pos=None):
    if x <= 0:
        return "0"
    e = int(math.floor(math.log10(x) + 1e-9))
    m = x / 10 ** e
    return rf"${m:g}\times10^{{{e}}}$" if abs(m - 1) > 1e-9 else rf"$10^{{{e}}}$"


def panel(ax, label):
    ax.text(-0.15, 1.02, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))          # vector, for journal submission
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=600)  # raster, embedded in the HTML->PDF
    plt.close(fig)


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def read(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


S = {}
for row in read("strict_tables.csv"):
    if row["input"] == "seed":
        S[(row["rule"], int(row["r"]))] = row


def strict(rule, key, widths):
    return [int(S[(rule, r)][key]) for r in widths]


# default mode (zero multipliers dropped), seed = r, article v4 tables 2, 3, 5
DEF_R = [256, 288, 320, 336, 352, 384, 512, 640, 768, 1024]
DEF_AD = [505746, 640069, 790223, 603133, 472178, 561945, 1021838, 1596627, 1723185, 1668994]
DEF_C1 = [393092, 497471, 614191, 677185, 743188, 884483, 1572716, 2457371, 3538701, 6291256]

# ------------------------------------------------------------------ Figure 1: depth
R = sorted(r for (rule, r) in S if rule == "adaptive" and r >= 128)
ad, c1 = strict("adaptive", "depth", R), strict("chunk1", "depth", R)

fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.7))
a.plot(R, ad, "-o", color=BLUE, label="adaptive chunk, strict")
a.plot(R, c1, "-s", color=ORANGE, label="chunk size 1, strict")
a.plot(DEF_R, DEF_AD, "--", color=BLUE, lw=0.7, alpha=0.6, label="adaptive, zero-pruned (value-dependent)")
a.plot(DEF_R, DEF_C1, "--", color=ORANGE, lw=0.7, alpha=0.6, label="chunk 1, zero-pruned (value-dependent)")
log2_axis(a, [128, 256, 512, 1024])
a.set_yscale("log")
a.yaxis.set_major_formatter(mticker.FuncFormatter(sci))
a.set_xlabel("block width $r$")
a.set_ylabel("ASAP depth (measured)")
a.legend(frameon=False, loc="upper left")
panel(a, "(a)")

b.axhline(1.0, color="#444444", lw=0.6, ls=":")
b.plot(R, [x / y for x, y in zip(ad, c1)], "-o", color=VERM, label="strict")
b.plot(DEF_R, [x / y for x, y in zip(DEF_AD, DEF_C1)], "--", color=VERM, lw=0.7, alpha=0.6,
       label="zero-pruned (value-dependent)")
log2_axis(b, [128, 256, 512, 1024])
b.set_xlabel("block width $r$")
b.set_ylabel("depth ratio, adaptive / chunk 1")
b.set_ylim(0.1, 1.45)
b.text(0.03, 0.04, "below 1: adaptive has smaller depth", transform=b.transAxes, fontsize=6.8,
       color="#444444")
b.text(0.03, 0.93, "above 1: chunk 1 has smaller depth", transform=b.transAxes, fontsize=6.8,
       color="#444444", va="top")
b.legend(frameon=False, loc="center right")
panel(b, "(b)")
fig.tight_layout(w_pad=2.2)
save(fig, "fig1_depth")

# ------------------------------------------------------------------ Figure 2: cost in real flops
RO = [64, 128, 256, 512, 1024]
word = [8 * int(S[("adaptive", r)]["shears"]) + 6 * int(S[("adaptive", r)]["scales"]) for r in RO]
chain = [2 * w + 18 * r for w, r in zip(word, RO)]
fft = [5 * r * math.log2(r) for r in RO]
toep_fft = [2 * 5 * (2 * r) * math.log2(2 * r) + 12 * r for r in RO]
direct_t = [4 * r * (r + 1) for r in RO]

fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.7))
a.plot(RO, chain, "-o", color=VERM, label="DFT by the word (two Toeplitz words + diagonals)")
a.plot(RO, word, "-D", color=PURPLE, label="one Toeplitz word")
a.plot(RO, direct_t, "--s", color=BLACK, lw=0.8, label=r"direct triangular Toeplitz, $4r(r+1)$")
a.plot(RO, toep_fft, ":v", color=SKY, label="Toeplitz via circulant FFT")
a.plot(RO, fft, ":^", color=GREEN, label=r"radix-2 FFT, $5r\log_2 r$")
log2_axis(a, RO)
a.set_yscale("log")
a.set_ylim(1e3, 1e11)
a.yaxis.set_major_formatter(mticker.FuncFormatter(sci))
a.set_xlabel("block width $r$")
a.set_ylabel("real floating-point operations")
a.legend(frameon=False, loc="upper left", fontsize=6.3)
panel(a, "(a)")

b.plot(RO, [c / f for c, f in zip(chain, fft)], "-o", color=VERM, label="DFT by the word / radix-2 FFT")
b.plot(RO, [w / t for w, t in zip(word, toep_fft)], "-D", color=PURPLE,
       label="Toeplitz word / circulant FFT")
b.plot(RO, [w / d for w, d in zip(word, direct_t)], "--s", color=BLACK, lw=0.8,
       label="Toeplitz word / direct Toeplitz")
log2_axis(b, RO)
b.set_yscale("log")
b.set_ylim(10, 1e5)
b.yaxis.set_major_formatter(mticker.FuncFormatter(sci))
b.set_xlabel("block width $r$")
b.set_ylabel("cost ratio (same units)")
b.legend(frameon=False, loc="upper left", fontsize=6.3)
panel(b, "(b)")
fig.tight_layout(w_pad=2.2)
save(fig, "fig2_cost")

# ------------------------------------------------------------------ Figure 3: log^4 model
M = read("depth_model_strict.csv")
e_vals = [int(m["e"]) for m in M]
norm = [float(m["ratio"]) for m in M]
fig, a = plt.subplots(figsize=(W1, 2.6))
a.plot([2 ** e for e in e_vals], norm, "-o", color=BLUE)
a.axhspan(min(norm), max(norm), color=BLUE, alpha=0.08, lw=0)
a.set_xscale("log", base=2)
a.set_xticks([2 ** e for e in e_vals])
a.set_xticklabels([rf"$2^{{{e}}}$" for e in e_vals])
a.xaxis.set_minor_formatter(mticker.NullFormatter())
a.set_ylim(0, 1.15 * max(norm))
a.set_xlabel("block width $r$ (model, not measured)")
a.set_ylabel(r"$d_{\mathrm{model}}(r)\,/\,\log_2^4 r$")
a.text(2 ** e_vals[0], 1.07 * max(norm), f"range {min(norm):.0f}–{max(norm):.0f}", fontsize=7,
       color="#333333")
fig.tight_layout()
save(fig, "fig3_log4_law")

# ------------------------------------------------------------------ Figure 4: trade-off at r = 512
fig, a = plt.subplots(figsize=(W1, 2.7))
pts = [("chunk1", "chunk size 1", ORANGE, "s"), ("cap2", "cap 2 (dominated)", BLACK, "^"),
       ("adaptive", "adaptive, b = 4", BLUE, "o")]
for key, lab, col, mk in pts:
    row = S[(key, 512)]
    x, y = int(row["ops"]), int(row["depth"])
    a.scatter(x, y, marker=mk, s=30, color=col, zorder=3)
    a.annotate(lab, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=7)
a.set_xscale("log")
a.set_yscale("log")
a.xaxis.set_major_formatter(mticker.FuncFormatter(sci))
a.yaxis.set_major_formatter(mticker.FuncFormatter(sci))
a.set_xticks([3e6, 1e7, 3e7])
a.set_yticks([1e6, 2e6, 3e6])
a.xaxis.set_minor_formatter(mticker.NullFormatter())
a.yaxis.set_minor_formatter(mticker.NullFormatter())
a.set_xlim(2.5e6, 6e7)
a.set_ylim(8e5, 3.2e6)
a.set_xlabel("gates (shears + scales)")
a.set_ylabel("ASAP depth")
a.grid(True, which="major", alpha=0.2, lw=0.5)
fig.tight_layout()
save(fig, "fig4_tradeoff_r512")

# ------------------------------------------------------------------ Figure 5: precision
P = read("dft_precision.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.7))
# full DFT chain in mp (v6: all widths 7..256; the lower of the two precisions per width)
MPD = {}
for p in read(os.path.join("v6", "mp_dft.csv")):
    r, dps = int(p["r"]), int(p["dps"])
    if r not in MPD or dps < MPD[r][0]:
        MPD[r] = (dps, float(p["digits_lost"]))
mp_dft = sorted((r, v[1]) for r, v in MPD.items())
f64 = sorted({int(p["r"]): float(p["max_rel_err"]) for p in P if p["kind"] == "f64_dft"}.items())
toe_g = {int(p["r"]): float(p["log10_growth"]) for p in P if p["kind"] == "mp_toeplitz"}
TOE = {}
for p in read(os.path.join("v6", "toeplitz.csv")):
    r, dps = int(p["r"]), int(p["dps"])
    if r not in TOE or dps < TOE[r][0]:
        TOE[r] = (dps, float(p["digits_lost"]))
toe = sorted((r, (TOE[r][1] if r in TOE else d, toe_g[r]))
             for r, d in ((int(p["r"]), float(p["digits_lost"])) for p in P if p["kind"] == "mp_toeplitz"))
a.plot([r for r, _ in mp_dft], [d for _, d in mp_dft], "-o", color=VERM,
       label="full DFT chain, $u=e^{2\\pi i/r}$")
rr = [2 ** (k / 8) for k in range(int(8 * math.log2(6)), int(8 * math.log2(300)) + 1)]
CL2 = 1.0149416064096536   # Clausen function Cl_2(pi/3)
a.plot(rr, [r / math.pi * CL2 / math.log(10) for r in rr], "--", color=BLACK, lw=0.8,
       label=r"main term $c\,r,\ c=\mathrm{Cl}_2(\pi/3)/(\pi\ln 10)$")
a.plot([r for r, _ in toe], [d for _, (d, _) in toe], "-s", color=PURPLE,
       label="Toeplitz word, random $h$ (strict)")
a.plot([r for r, _ in toe], [g for _, (_, g) in toe], ":", color=PURPLE, marker="x",
       label=r"$\log_{10}\max_k |g_k|$ for the same $h$")
log2_axis(a, [8, 32, 128, 512])
a.set_xlim(6, 1400)
a.set_xlabel("block width $r$")
a.set_ylabel("decimal digits lost (mp)")
a.legend(frameon=False, loc="upper left")
panel(a, "(a)")

b.plot([r for r, _ in f64], [e for _, e in f64], "-o", color=VERM)
b.axhline(1.0, color="#444444", lw=0.6, ls=":")
log2_axis(b, [8, 16, 32, 64, 96])
b.set_yscale("log")
b.set_yticks([1e-15, 1e0, 1e15, 1e30, 1e45, 1e60])
b.yaxis.set_minor_locator(mticker.NullLocator())
b.yaxis.set_major_formatter(mticker.FuncFormatter(sci))
b.set_ylim(1e-17, 1e62)
b.set_xlabel("block width $r$")
b.set_ylabel("max relative error, float64")
b.text(0.03, 0.93, "full DFT chain; $\\kappa_2(F_r)=1$", transform=b.transAxes, fontsize=7,
       va="top", color="#444444")
panel(b, "(b)")
fig.tight_layout(w_pad=2.2)
save(fig, "fig5_precision")

# ------------------------------------------------------------------ Figure 6: stage-wise loss
SW = read(os.path.join("v6", "stagewise.csv"))
stages = ["$D_H$", r"$T^{\mathsf{T}}$", r"$D_{\mathrm{mid}}$", "$T$", "$D_H$"]
widths = sorted({int(s["r"]) for s in SW})
fig, a = plt.subplots(figsize=(W1, 2.7))
bw = 0.8 / len(widths)
cols = [SKY, BLUE, BLACK]
for wi, r in enumerate(widths):
    vals = [max(0.0, float(s["local_lost"])) for s in sorted(
        (s for s in SW if int(s["r"]) == r), key=lambda s: int(s["stage"]))]
    xs = [k + (wi - (len(widths) - 1) / 2) * bw for k in range(5)]
    a.bar(xs, vals, width=bw * 0.95, color=cols[wi % len(cols)], label=f"$r={r}$")
a.set_xticks(range(5))
a.set_xticklabels(stages)
a.set_ylabel("digits lost by the stage alone")
a.set_xlabel("stage of the chain (applied left to right)")
a.legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig6_stagewise")

print("wrote fig1..fig6")
