#!/usr/bin/env python3
"""UV-Vis kinetics grapher: overlaid spectra + rate-law plot from a series of text spectra.

GUI:  python uvvis_kinetics.py
CLI:  python uvvis_kinetics.py --cli FILES... --title "20261017 Bi-Ni dark" \
          --reactant 400 --baseline 850 --order 1 --window 0.5 --out name

Input : 2-column text (wavelength nm, absorbance), optional header ending in
        '>>>>>Begin Spectral Data<<<<<'. Time (min) is read from the file name
        ('2.5 min' / '2_5_min'), or set by hand (GUI: double-click a file;
        CLI: --times 0,2.5,5,...  one value per file, in the order given).
Output: <out>_spectra.png (15 x 10 cm, 3:2), <out>_kinetics.png (15 x 9.375 cm, 8:5),
        both 600 dpi, same width and same text sizes (Arial),
        plus <out>_kinetics.csv (per-point values and fit results).
Needs : numpy, matplotlib (+ tkinter for the GUI).

Rate laws (y vs time; slope = k_app):
  order 0: y = A      order 1: y = ln A      order 2: y = 1/A
  "zero at t=0" subtracts the first point so each series starts at 0.
  Fit is forced through the origin by default (y = k t); --free-intercept fits y = k t + b.
  R^2 uses the usual 1 - SS_res/SS_tot (can be negative for a through-origin fit).

Baseline handling (two options, do not combine):
  --baseline NM : kinetics only; per file, A(peak) - A(NM)  (e.g. 400 - 850 nm).
  --sub-nm NM   : same per-file subtraction applied to the WHOLE spectrum (both graphs).
  --sub-file F  : subtract a blank/reference spectrum from every spectrum.
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import matplotlib
import matplotlib.ticker
from matplotlib.figure import Figure

# ------------------------------------------------------------------ constants
# Arial everywhere (falls back to the metric-identical Liberation Sans, then Helvetica /
# DejaVu if Arial is not installed). Math text (ln(A/A0) etc.) uses the same family.
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "mathtext.cal": "sans", "mathtext.tt": "sans", "mathtext.sf": "sans",
})

CM = 1 / 2.54                          # inches per cm
EXPORT_W = 15 * CM                     # both graphs export at this width
SPEC_SIZE = (EXPORT_W, 10 * CM)        # spectra  3:2 -> 15 x 10 cm
KIN_SIZE = (EXPORT_W, EXPORT_W * 5 / 8)  # kinetics 8:5 -> 15 x 9.375 cm
# One set of text sizes (pt) for BOTH graphs, so text looks identical at equal width.
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 10, 9, 8, 7
SAVE_DPI = 600                     # output resolution
PREVIEW_DPI = 100                  # GUI preview start dpi (auto-raised to fit the window)

SPEC_COLORS = ["#2f6fdc", "#f28e2b", "#d62d20", "#3cbf6e", "#f5c518",
               "#8a2be2", "#22d3d3", "#8c564b", "#e377c2", "#7f7f7f"]
KIN_COLORS = ["#4285f4", "#ea8a2e"]            # reactant, product
UNITS = {0: "a.u. min^-1", 1: "min^-1", 2: "a.u.^-1 min^-1"}   # k_app units per order
YLABELS = {(0, True): r"$A - A_0$ (a.u.)", (0, False): "A (a.u.)",
           (1, True): r"ln($A/A_0$)", (1, False): "ln A",
           (2, True): r"$1/A - 1/A_0$ (a.u.$^{-1}$)", (2, False): r"1/A (a.u.$^{-1}$)"}


# ------------------------------------------------------------------ data loading
def parse_time(path):
    """Minutes from a file name ('... 2.5 min', '2_5_min'); None if absent.
    A leading instrument timestamp (10+ digits) is ignored."""
    name = re.sub(r"^\d{10,}[_ ]", "", os.path.basename(path))
    m = re.search(r"(?<![\d.])(\d+(?:[._]\d+)?)[\s_]*min", name, re.I)
    return float(m.group(1).replace("_", ".")) if m else None


def load_spectrum(path):
    """Read (wavelength, absorbance) arrays; skips any header up to the data marker."""
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    start = next((i + 1 for i, ln in enumerate(lines)
                  if ln.strip().startswith(">>>>>Begin")), 0)
    wl, ab = [], []
    for ln in lines[start:]:
        parts = ln.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            wl.append(float(parts[0]))
            ab.append(float(parts[1]))
        except ValueError:          # stray text line
            continue
    return np.array(wl), np.array(ab)


def load_series(entries):
    """entries: [(path, time_min)] -> [(t, wl, abs)] sorted by time."""
    out = []
    for path, t in entries:
        if t is None:
            raise ValueError(f"No time for {os.path.basename(path)} "
                             "(put '<n> min' in the name or set it by hand)")
        wl, ab = load_spectrum(path)
        out.append((t, wl, ab))
    out.sort(key=lambda s: s[0])
    return out


# ------------------------------------------------------------------ absorbance reading
def peak_abs(wl, ab, center, half_window=2.0, track=False, good=None):
    """Absorbance at `center`: median of points within +/- half_window nm
    (max instead if track=True). Exact-zero points are instrument glitches and are
    ignored (`good` can override that mask)."""
    if good is None:
        good = ab != 0
    sel = good & (np.abs(wl - center) <= half_window)
    if not sel.any():
        raise ValueError(f"No data near {center:g} nm")
    v = ab[sel]
    return float(v.max() if track else np.median(v))


def subtract(series, nm=None, ref_path=None, half_window=2.0):
    """Normalize whole spectra before plotting/fitting.
    ref_path: subtract a blank/reference spectrum (interpolated onto each wavelength grid).
    nm      : subtract each spectrum's own A at nm (flat offset, e.g. 850).
    Glitch (exact-zero) points stay zero so later steps keep ignoring them."""
    if nm is None and ref_path is None:
        return series
    rwl, rab = load_spectrum(ref_path) if ref_path else (None, None)
    out = []
    for t, wl, ab in series:
        good = ab != 0
        y = ab - np.interp(wl, rwl, rab) if ref_path else ab.copy()
        if nm is not None:
            y -= peak_abs(wl, y, nm, half_window, good=good)
        y[~good] = 0
        out.append((t, wl, y))
    return out


# ------------------------------------------------------------------ kinetics
def kinetics(series, wl_nm, order=1, zero=True, half_window=2.0, baseline_nm=None,
             track=False, fit_range=None, through_zero=True):
    """Rate-law fit at one wavelength. Returns dict: t, A, y, mask (points in fit),
    k (slope), b (intercept), r2."""
    t = np.array([s[0] for s in series], float)
    A = np.array([peak_abs(s[1], s[2], wl_nm, half_window, track) for s in series])
    if baseline_nm is not None:                       # per-file baseline subtraction
        A -= np.array([peak_abs(s[1], s[2], baseline_nm, half_window) for s in series])

    if order == 0:
        y = A.copy()
    elif order in (1, 2):
        if np.any(A <= 0):
            raise ValueError(f"A <= 0 at {wl_nm:g} nm: ln A and 1/A undefined")
        y = np.log(A) if order == 1 else 1.0 / A
    else:
        raise ValueError("order must be 0, 1 or 2")
    if zero:
        y = y - y[0]

    m = np.ones(len(t), bool)                         # which points enter the fit
    if fit_range:
        lo, hi = fit_range
        if lo is not None:
            m &= t >= lo
        if hi is not None:
            m &= t <= hi
    if m.sum() < 2:
        raise ValueError("fit range has fewer than 2 points")

    tm, ym = t[m], y[m]
    if through_zero:                                  # least squares for y = k t
        k, b = float(tm @ ym / (tm @ tm)), 0.0
    else:
        k, b = np.polyfit(tm, ym, 1)
    ss_tot = np.sum((ym - ym.mean()) ** 2)
    r2 = 1 - np.sum((ym - (k * tm + b)) ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return dict(t=t, A=A, y=y, mask=m, k=float(k), b=float(b), r2=float(r2))


def build_peaks(reactant, product):
    """[(label, nm)] for whichever peaks were given."""
    return [(lab, nm) for lab, nm in (("Reactant", reactant), ("Product", product))
            if nm is not None]


# ------------------------------------------------------------------ plotting
def make_spectra_fig(series, title, wmin, wmax, ymin=None, ymax=None, fig=None):
    """Overlay of all spectra, 15 x 10 cm. Reuses/clears `fig` if given (GUI preview)."""
    fig = fig or Figure(figsize=SPEC_SIZE, dpi=SAVE_DPI)
    fig.clf()
    ax = fig.add_subplot(111)

    n = len(series)
    if n <= len(SPEC_COLORS):
        cols = SPEC_COLORS[:n]
    else:                                             # many spectra: continuous colormap
        cm = matplotlib.colormaps["turbo"]
        cols = [cm(i / (n - 1)) for i in range(n)]

    lo, hi = np.inf, -np.inf                          # data extent inside the x range
    for (t, wl, ab), c in zip(series, cols):
        sel = (wl >= wmin) & (wl <= wmax) & (ab != 0)
        ax.plot(wl[sel], ab[sel], lw=0.8, color=c, label=f"{t:g} min")
        if sel.any():
            lo, hi = min(lo, ab[sel].min()), max(hi, ab[sel].max())

    # default y range: starts at 0 (lower if data go negative), top rounded up to 0.2
    if ymin is None:
        ymin = min(0.0, np.floor(lo * 5) / 5) if np.isfinite(lo) else 0.0
    if ymax is None:
        ymax = np.ceil(hi * 1.02 * 5) / 5 if np.isfinite(hi) else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    ax.set_xlim(wmin, wmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.linspace(wmin, wmax, 5))
    ax.grid(color="#e3e3e3", lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel("Wavelength (nm)", fontsize=FS_LABEL)
    ax.set_ylabel("Absorbance (a.u.)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    leg = ax.legend(loc="upper right", frameon=True, edgecolor="black",
                    fancybox=False, fontsize=FS_LEGEND)
    for txt, ln in zip(leg.get_texts(), ax.get_lines()):   # legend text in line colour
        txt.set_color(ln.get_color())
    fig.tight_layout()
    return fig


def make_kinetics_fig(series, title, peaks, order=1, zero=True, half_window=2.0,
                      baseline_nm=None, track=False, fit_range=None, show_k=True,
                      through_zero=True, fig=None):
    """Rate-law scatter + fit line, 15 x 9.375 cm (8:5). peaks: [(label, nm)].
    Returns (fig, [result dicts with label/wl added])."""
    fig = fig or Figure(figsize=KIN_SIZE, dpi=SAVE_DPI)
    fig.clf()
    ax = fig.add_subplot(111)

    results = []
    for j, (label, wl_nm) in enumerate(peaks):
        r = kinetics(series, wl_nm, order, zero, half_window, baseline_nm, track,
                     fit_range, through_zero)
        r.update(label=label, wl=wl_nm)
        results.append(r)

        c = KIN_COLORS[j % len(KIN_COLORS)]
        lab = f"{wl_nm:g} nm" + (f", k={r['k']:.3g}, $R^2$={r['r2']:.3f}" if show_k else "")
        t, y, m = r["t"], r["y"], r["mask"]
        ax.plot(t[m], y[m], "o", color=c, ms=5, label=lab, clip_on=False)  # points in the fit
        if not m.all():                                             # excluded: open circles
            ax.plot(t[~m], y[~m], "o", mfc="white", mec=c, ms=5, clip_on=False)
        tt = np.linspace(t[m].min(), t[m].max(), 2)                 # straight line: 2 pts
        ax.plot(tt, r["k"] * tt + r["b"], "--", color=c, lw=1.0)

    # Axes hug the data: x from first to last time, y from lowest to highest value, so
    # for a decay the t=0 dot is top-left and the last dot bottom-right (rising data:
    # bottom-left / top-right). clip_on=False above keeps edge dots un-cropped.
    t_all = np.concatenate([r["t"] for r in results])
    y_all = np.concatenate([r["y"] for r in results])
    ax.set_xlim(t_all.min(), t_all.max())
    # ticks: every sampled time when there are few (0, 2.5, ... 15), so the axis visibly
    # runs first -> last time; with many points keep automatic ticks but add both ends
    ts = np.unique(t_all)
    if len(ts) <= 8:
        ax.set_xticks(ts)
    else:
        ax.set_xticks(np.unique(np.r_[ax.get_xticks()[(ax.get_xticks() > ts[0]) &
                                                       (ax.get_xticks() < ts[-1])],
                                      ts[0], ts[-1]]))
    ax.set_xlim(ts[0], ts[-1])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    if y_all.max() > y_all.min():
        ax.set_ylim(y_all.min(), y_all.max())
    ax.set_title(f"{title} kinetics".strip(), loc="left", color="#555555", fontsize=FS_TITLE, pad=6)
    ax.set_xlabel("Time (min)", fontsize=FS_LABEL)
    ax.set_ylabel(YLABELS[(order, zero)], fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(color="#d9d9d9", lw=0.8)
    ax.set_axisbelow(True)
    for name, sp in ax.spines.items():
        if name in ("top", "right"):
            sp.set_visible(False)
        else:
            sp.set_linewidth(1.0)                       # same weight as the spectra frame
    ax.legend(loc="best", fontsize=FS_LEGEND, frameon=False, handletextpad=0.3,
              borderaxespad=0.3, labelspacing=0.3)
    fig.tight_layout(pad=0.4)   # tight outer margins
    return fig, results


def render(series, title, wmin, wmax, ymin, ymax, peaks, kin_kw, figs=(None, None)):
    """Build both graphs. kin_kw: keyword args for make_kinetics_fig (order, zero,
    half_window, baseline_nm, track, fit_range, show_k, through_zero).
    Returns (spectra_fig, kinetics_fig or None, results)."""
    spec = make_spectra_fig(series, title, wmin, wmax, ymin, ymax, figs[0])
    if not peaks:
        return spec, None, []
    kin, res = make_kinetics_fig(series, title, peaks, fig=figs[1], **kin_kw)
    return spec, kin, res


def write_csv(path, results, order):
    """Per-peak block: header row with k/intercept/R2, then time, A, y, in_fit."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for r in results:
            w.writerow([f"# {r['label']} {r['wl']:g} nm", "order", order,
                        "k_app", f"{r['k']:.6g}", UNITS[order],
                        "intercept", f"{r['b']:.6g}", "R2", f"{r['r2']:.5f}"])
            w.writerow(["time_min", "A", "y", "in_fit"])
            for t, A, y, m in zip(r["t"], r["A"], r["y"], r["mask"]):
                w.writerow([f"{t:g}", f"{A:.5f}", f"{y:.6f}", int(m)])
            w.writerow([])


def summary(results, order):
    """One text line per peak: k_app and R^2."""
    return "\n".join(f"{r['label']} {r['wl']:g} nm: k_app = {r['k']:.4g} {UNITS[order]}, "
                     f"R2 = {r['r2']:.4f}" for r in results)


# ------------------------------------------------------------------ CLI
def run_cli(a):
    if a.times:
        times = [float(x) for x in a.times.split(",")]
        if len(times) != len(a.files):
            sys.exit("--times needs one value per file")
        entries = list(zip(a.files, times))
    else:
        entries = [(p, parse_time(p)) for p in a.files]

    series = subtract(load_series(entries), a.sub_nm, a.sub_file, a.window)
    kin_kw = dict(order=a.order, zero=not a.no_zero, half_window=a.window,
                  baseline_nm=a.baseline, track=a.track,
                  fit_range=(a.fit_tmin, a.fit_tmax), show_k=True,
                  through_zero=not a.free_intercept)
    spec, kin, res = render(series, a.title, a.wmin, a.wmax, a.ymin, a.ymax,
                            build_peaks(a.reactant, a.product), kin_kw)

    spec.savefig(a.out + "_spectra.png", dpi=a.dpi)
    print("saved", a.out + "_spectra.png")
    if kin is not None:
        kin.savefig(a.out + "_kinetics.png", dpi=a.dpi)
        write_csv(a.out + "_kinetics.csv", res, a.order)
        print("saved", a.out + "_kinetics.png", a.out + "_kinetics.csv")
        print(summary(res, a.order))


# ------------------------------------------------------------------ GUI
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("UV-Vis Kinetics")
    entries = []                                      # [[path, time_min], ...]
    last = {}                                         # inputs of the last successful Plot
    subfile = {"p": None}                             # reference spectrum to subtract

    # ---- input variables
    def sv(d=""):
        return tk.StringVar(value=d)

    v = dict(title=sv(), reactant=sv(), product=sv(), wmin=sv("220"), wmax=sv("900"),
             ymin=sv(), ymax=sv(), window=sv("2"), subnm=sv(), baseline=sv(),
             tmin=sv(), tmax=sv(), order=sv("1 (ln A)"))
    zero = tk.BooleanVar(value=True)
    track = tk.BooleanVar(value=False)
    showk = tk.BooleanVar(value=True)
    free = tk.BooleanVar(value=False)
    status, refname = sv(), sv("none")

    def fl(key):
        """Entry -> float, or None if blank."""
        s = v[key].get().strip()
        return float(s) if s else None

    # ---- layout: controls left, graphs right
    left = ttk.Frame(root, padding=8)
    left.grid(row=0, column=0, sticky="ns")
    right = ttk.Frame(root)
    right.grid(row=0, column=1, sticky="nsew")
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    # ---- file list
    lb = tk.Listbox(left, width=44, height=9)
    lb.grid(row=1, column=0, columnspan=2)

    def refresh():
        entries.sort(key=lambda e: (e[1] is None, e[1] or 0))
        lb.delete(0, "end")
        for p, t in entries:
            lb.insert("end", f"{'?' if t is None else f'{t:g}'} min | {os.path.basename(p)}")

    def add_files():
        have = {e[0] for e in entries}
        for p in filedialog.askopenfilenames(filetypes=[("Text", "*.txt"), ("All", "*.*")]):
            if p not in have:
                entries.append([p, parse_time(p)])
        refresh()

    def remove_sel():
        for i in reversed(lb.curselection()):
            del entries[i]
        refresh()

    def edit_time(_event=None):
        sel = lb.curselection()
        if not sel:
            return
        val = simpledialog.askfloat("Time", "Time (min):",
                                    initialvalue=entries[sel[0]][1], parent=root)
        if val is not None:
            entries[sel[0]][1] = val
            refresh()

    lb.bind("<Double-Button-1>", edit_time)
    bar = ttk.Frame(left)                             # Add / Remove / Clear all
    bar.grid(row=0, column=0, columnspan=2)
    ttk.Button(bar, text="Add files...", command=add_files).pack(side="left")
    ttk.Button(bar, text="Remove selected", command=remove_sel).pack(side="left")
    ttk.Label(left, text="double-click a file to edit its time").grid(
        row=2, column=0, columnspan=2)

    # ---- parameter entries
    rows = [("Title", "title"), ("Reactant peak (nm)", "reactant"),
            ("Product peak (nm)", "product"), ("Wavelength min (nm)", "wmin"),
            ("Wavelength max (nm)", "wmax"), ("Abs min (optional)", "ymin"),
            ("Abs max (optional)", "ymax"), ("Peak window +/- nm", "window"),
            ("Subtract A at nm (all)", "subnm"), ("Baseline nm (kinetics)", "baseline"),
            ("Fit from t (min)", "tmin"), ("Fit to t (min)", "tmax")]
    r0 = 3
    for i, (lab, key) in enumerate(rows, start=r0):
        ttk.Label(left, text=lab).grid(row=i, column=0, sticky="w")
        ttk.Entry(left, textvariable=v[key], width=20).grid(row=i, column=1)
    i = r0 + len(rows)
    ttk.Label(left, text="Rate order (y vs t)").grid(row=i, column=0, sticky="w")
    ttk.Combobox(left, textvariable=v["order"], width=17, state="readonly",
                 values=["0 (A)", "1 (ln A)", "2 (1/A)"]).grid(row=i, column=1)
    for r, (txt, var, col) in enumerate([("Zero at t=0", zero, 0),
                                         ("Track max in window", track, 1),
                                         ("Show k, R\u00b2 on plot", showk, 0),
                                         ("Free intercept", free, 1)]):
        ttk.Checkbutton(left, text=txt, variable=var).grid(
            row=i + 1 + r // 2, column=col, sticky="w")

    def pick_ref():
        p = filedialog.askopenfilename(title="Reference/blank spectrum to subtract")
        subfile["p"] = p or None
        refname.set(os.path.basename(p) if p else "none")

    ttk.Button(left, text="Subtract file...", command=pick_ref).grid(row=i + 3, column=0)
    ttk.Label(left, textvariable=refname).grid(row=i + 3, column=1, sticky="w")

    # ---- preview figures at their true output size (canvas is not stretched, so the
    # preview layout matches the saved file exactly)
    fig1 = Figure(figsize=SPEC_SIZE, dpi=PREVIEW_DPI)
    fig2 = Figure(figsize=KIN_SIZE, dpi=PREVIEW_DPI)
    nb = ttk.Notebook(right)
    nb.pack(fill="both", expand=True)
    canvases = []

    def fit_to_tab(fig, cv, size_in):
        """Return a <Configure> handler that raises the preview dpi so the graph fills
        its tab while keeping the export aspect ratio (3:2 spectra, 8:5 kinetics).
        Only the on-screen preview changes; saved files are rendered fresh at exact cm."""
        def on_resize(event):
            dpi = min((event.width - 16) / size_in[0], (event.height - 16) / size_in[1])
            if dpi < 40 or abs(dpi - fig.dpi) < 3:      # tiny/unchanged: ignore (no loops)
                return
            fig.set_dpi(dpi)
            cv.get_tk_widget().configure(width=int(size_in[0] * dpi),
                                         height=int(size_in[1] * dpi))
            cv.draw_idle()
        return on_resize

    for fig, name, size in ((fig1, "Spectra", SPEC_SIZE), (fig2, "Kinetics", KIN_SIZE)):
        tab = ttk.Frame(nb)
        nb.add(tab, text=name)
        cv = FigureCanvasTkAgg(fig, tab)
        cv.get_tk_widget().pack(anchor="nw", padx=6, pady=6)
        tab.bind("<Configure>", fit_to_tab(fig, cv, size))
        canvases.append(cv)

    def clear_previews():
        """Blank both graphs so nothing from a previous Plot (or failed Plot) lingers."""
        for fig, cv in ((fig1, canvases[0]), (fig2, canvases[1])):
            fig.clf()
            cv.draw()

    def clear_all():
        """Remove every loaded spectrum and blank both graphs (settings are kept)."""
        entries.clear()
        refresh()
        clear_previews()
        status.set("")
        last.clear()

    ttk.Button(bar, text="Clear all", command=clear_all).pack(side="left")

    def plot():
        clear_previews()
        status.set("")
        last.clear()
        try:
            if not entries:
                raise ValueError("Add files first")
            win = fl("window") or 2.0
            series = subtract(load_series(entries), fl("subnm"), subfile["p"], win)
            order = int(v["order"].get()[0])
            kin_kw = dict(order=order, zero=zero.get(), half_window=win,
                          baseline_nm=fl("baseline"), track=track.get(),
                          fit_range=(fl("tmin"), fl("tmax")), show_k=showk.get(),
                          through_zero=not free.get())
            args = (series, v["title"].get(), fl("wmin"), fl("wmax"), fl("ymin"),
                    fl("ymax"), build_peaks(fl("reactant"), fl("product")), kin_kw)
            _, kin, res = render(*args, figs=(fig1, fig2))
            canvases[0].draw()
            if kin is not None:
                canvases[1].draw()
                status.set(summary(res, order))
            last.update(args=args, res=res, order=order)   # remembered for Save
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save():
        if not last:
            messagebox.showinfo("Nothing to save", "Click Plot first.")
            return
        base = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=(v["title"].get() or "plot"))
        if not base:
            return
        base = os.path.splitext(base)[0]
        spec, kin, _ = render(*last["args"])          # fresh figures at exact cm size
        spec.savefig(base + "_spectra.png", dpi=SAVE_DPI)
        saved = [base + "_spectra.png"]
        if kin is not None:
            kin.savefig(base + "_kinetics.png", dpi=SAVE_DPI)
            write_csv(base + "_kinetics.csv", last["res"], last["order"])
            saved += [base + "_kinetics.png", base + "_kinetics.csv"]
        messagebox.showinfo("Saved", "\n".join(saved))

    ttk.Button(left, text="Plot", command=plot).grid(row=i + 4, column=0, pady=6)
    ttk.Button(left, text="Save PNGs + CSV", command=save).grid(row=i + 4, column=1)
    ttk.Label(left, textvariable=status, justify="left", wraplength=300).grid(
        row=i + 5, column=0, columnspan=2, sticky="w")
    root.geometry("1300x800")     # fixed start size so the preview can't grow the window
    root.minsize(900, 600)
    root.mainloop()


# ------------------------------------------------------------------ entry point
if __name__ == "__main__":
    if "--cli" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--cli", action="store_true")
        ap.add_argument("files", nargs="+")
        ap.add_argument("--times", help="comma list of minutes, one per file, in file order")
        ap.add_argument("--title", default="")
        ap.add_argument("--reactant", type=float, help="reactant peak, nm")
        ap.add_argument("--product", type=float, help="product peak, nm")
        ap.add_argument("--wmin", type=float, default=220)
        ap.add_argument("--wmax", type=float, default=900)
        ap.add_argument("--ymin", type=float)
        ap.add_argument("--ymax", type=float)
        ap.add_argument("--order", type=int, default=1, choices=[0, 1, 2])
        ap.add_argument("--no-zero", action="store_true", help="do not subtract first point")
        ap.add_argument("--track", action="store_true", help="max in window instead of median")
        ap.add_argument("--window", type=float, default=2.0, help="peak window +/- nm")
        ap.add_argument("--baseline", type=float, help="kinetics only: subtract A at this nm")
        ap.add_argument("--sub-nm", type=float,
                        help="subtract each spectrum's A at this nm from the WHOLE spectrum "
                             "(both graphs), e.g. 850")
        ap.add_argument("--sub-file", help="subtract this reference/blank spectrum file")
        ap.add_argument("--fit-tmin", type=float)
        ap.add_argument("--fit-tmax", type=float)
        ap.add_argument("--free-intercept", action="store_true",
                        help="fit y = k t + b instead of through the origin")
        ap.add_argument("--dpi", type=int, default=SAVE_DPI)
        ap.add_argument("--out", default="plot")
        run_cli(ap.parse_args())
    else:
        run_gui()
