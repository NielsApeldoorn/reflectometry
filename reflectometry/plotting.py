"""
Plotting helpers. Each function returns the figure(s) it creates and does not
call ``plt.show()``, so callers decide when and how to display or save them.
"""

import matplotlib.pyplot as plt
import numpy as np

from .cable import CableParams, cable_model
from .circuit import BridgeParams, bridge_matrix, cascade
from .experimental import MeasuredResonances
from .vna import VnaTrace
from .reflection import load_impedance, reflection_db
from .analysis import flatten_minima, default_frequencies, reflection_spectrum_db, resonance_sweep
from .vna import find_trace_minima

DISPLAY_UNITS = {
    # name:   (scale applied to the SI value, unit shown in labels)
    "length": (1.0, "m"),
    "Z0c":    (1.0, "ohm"),
    "VF":     (1.0, ""),
    "R_T":    (1.0, "ohm"),
    "C_T":    (1e12, "pF"),
    "L":      (1e6, "uH"),
    "R_L":    (1.0, "ohm"),
    "C_P":    (1e12, "pF"),
    "R_eq":   (1.0, "ohm"),
}


def format_params(params, units=DISPLAY_UNITS):
    """'R_T=1000 ohm, L=0.47 uH' from a params dict. Unknown keys show the raw value."""
    parts = []
    for k, v in params.items():
        scale, unit = units.get(k, (1.0, ""))
        parts.append(f"{k}={v*scale:g}" + (f" {unit}" if unit else ""))
    return ", ".join(parts)


def plot_cable_effect(f, cable=CableParams(), bridge=BridgeParams()):
    """
    Load impedance magnitude and reflection with the cable, without it, and
    with the cable alone terminated by ``bridge.R_eq``.

    Returns
    -------
    (fig_impedance, fig_reflection)
    """
    abcd, _, _ = cable_model(f, cable)
    Load_matrix = bridge_matrix(f, bridge)
    R_eq = bridge.R_eq

    Z_load = load_impedance(Load_matrix, R_eq)
    Z_load_cable = load_impedance(cascade(abcd, Load_matrix), R_eq)
    Z_load_only_cable = load_impedance(abcd, R_eq)

    fig1, ax1 = plt.subplots()
    ax1.semilogx(f, np.abs(Z_load_cable), label="With cable")
    ax1.semilogy(f, np.abs(Z_load), label="Without cable")
    ax1.semilogy(f, np.abs(Z_load_only_cable), label="Only cable")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("|Z| (ohm)")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.semilogx(f, reflection_db(Z_load_cable, cable.Z0c), label="With cable")
    ax2.semilogx(f, reflection_db(Z_load, cable.Z0c), label="Without cable")
    ax2.semilogx(f, reflection_db(Z_load_only_cable, cable.Z0c), label="Only cable")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Reflection (dB)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend()

    return fig1, fig2


def plot_resonance_vs_impedance(sweep, measured=None, title="Frequency and value vs. impedance",
                                save_path=None):
    """
    Model resonance (frequency and dB depth) against load resistance, optionally
    overlaid with measured points.

    Parameters
    ----------
    sweep : ResonanceSweep
    measured : MeasuredResonances, optional
    title : str
    save_path : str, optional
        If given, the figure is saved to this path at 150 dpi.

    Returns
    -------
    fig
    """
    color_f = "tab:blue"
    color_db = "tab:red"

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    if measured is not None:
        z, f_mhz, db = measured.sorted_by_impedance()
        ax1.plot(z, f_mhz, marker="o", color=color_f, label="Frequency")

    ax1.plot(sweep.resistances, sweep.resonance_frequencies * 1e-6,
             linestyle="--", color="tab:green", label="Resonance Frequency")
    ax1.set_xlabel("Impedance (Ω)")
    ax1.set_ylabel("Frequency (MHz)", color=color_f)
    ax1.tick_params(axis="y", labelcolor=color_f)
    ax1.set_xscale("log")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    if measured is not None:
        ax2.plot(z, db, marker="s", color=color_db, label="Value (dB)")
    ax2.plot(sweep.resistances, sweep.gamma_at_resonance_db,
             linestyle="--", color="tab:orange", label="Gamma at Resonance (dB)")
    ax2.set_ylabel("Value (dB)", color=color_db)
    ax2.tick_params(axis="y", labelcolor=color_db)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower right", fontsize=10)

    ax1.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_parametric(points, value_label=None, quantity="frequency", title=None):
    """
    Overlay the resonance curves from a ``parametric_sweep`` result.

    Parameters
    ----------
    points : list of ParametricPoint
    value_label : callable, optional
        Maps a ``params`` dict to a legend label. Defaults to ``str(params)``.
    quantity : {"frequency", "gamma"}
        Which quantity to plot against load resistance.
    title : str, optional

    Returns
    -------
    fig
    """
    if value_label is None:
        value_label = format_params

    fig, ax = plt.subplots(figsize=(8, 5))
    for pt in points:
        mask = pt.result 
        R, F, G = flatten_minima(pt.result)

        if quantity == "frequency":
            y = F * 1e-6
            ax.scatter(R, y, s=3, label=value_label(pt.params))
        elif quantity == "gamma":
            y = G
            ax.scatter(R, y, s=3, label=value_label(pt.params))
        else:
            raise ValueError("quantity must be 'frequency' or 'gamma'")
        

    ax.set_xscale("log")
    ax.set_xlabel("Load resistance (ohm)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_spectra(points, resistance, f=None, value_label=None, show_minima=True):
    """|Gamma| vs frequency for each point, at the sweep resistance nearest `resistance`."""
    if f is None:
        f = default_frequencies()
    if value_label is None:
        value_label = format_params

    fig, ax = plt.subplots(figsize=(8, 5))
    used = []
    for pt in points:
        j = int(np.argmin(np.abs(pt.result.resistances - resistance)))
        r_used = pt.result.resistances[j]
        used.append(r_used)
        line, = ax.semilogx(f, reflection_spectrum_db(f, r_used, pt.cable, pt.bridge),
                            label=value_label(pt.params))
        if show_minima:
            fk = np.atleast_1d(pt.result.resonance_frequencies[j])
            gk = np.atleast_1d(pt.result.gamma_at_resonance_db[j])
            ax.scatter(fk, gk, color=line.get_color(), marker="x", s=30, zorder=3)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|Γ| (dB)")
    ax.set_title(f"|Γ| vs frequency at R = {used[0]:g} ohm")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig

import math
import numpy as np


def _grid(points, x_key, y_key):
    xs = np.array(sorted({pt.params[x_key] for pt in points}))
    ys = np.array(sorted({pt.params[y_key] for pt in points}))
    cell_of, where = {}, {}
    for p, pt in enumerate(points):
        i = int(np.searchsorted(xs, pt.params[x_key]))
        j = int(np.searchsorted(ys, pt.params[y_key]))
        cell_of[p] = (j, i)
        where[(j, i)] = p
    return xs, ys, cell_of, where


def align_sorted(a, b, gap=0.3):
    """
    Align two sorted frequency arrays. A match costs the log-frequency distance;
    skipping a root on either side costs `gap`. Returns matched (i, j) pairs.
    """
    n, m = len(a), len(b)
    la, lb = np.log(a), np.log(b)
    D = np.zeros((n + 1, m + 1))
    D[1:, 0] = gap * np.arange(1, n + 1)
    D[0, 1:] = gap * np.arange(1, m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(la[i - 1] - lb[j - 1])
            D[i, j] = min(D[i - 1, j - 1] + c, D[i - 1, j] + gap, D[i, j - 1] + gap)

    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        c = abs(la[i - 1] - lb[j - 1])
        if abs(D[i, j] - (D[i - 1, j - 1] + c)) < 1e-12:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif abs(D[i, j] - (D[i - 1, j] + gap)) < 1e-12:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]

BIG = 1e6

def _ordered_match(a, b):
    """
    Match sorted arrays a and b in order, giving exactly min(len(a), len(b)) pairs
    with the smallest total log-frequency distance. The longer list may skip entries
    (a branch appears or disappears); the shorter list may not.
    Returns a list of (i, j) index pairs.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    la, lb = np.log(a), np.log(b)
    skip_a = 0.0 if n > m else BIG      # skipping an entry of a
    skip_b = 0.0 if m > n else BIG      # skipping an entry of b
    D = np.zeros((n + 1, m + 1))
    D[1:, 0] = np.cumsum([skip_a] * n)
    D[0, 1:] = np.cumsum([skip_b] * m)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(la[i - 1] - lb[j - 1])
            D[i, j] = min(D[i - 1, j - 1] + c, D[i - 1, j] + skip_a, D[i, j - 1] + skip_b)

    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        c = abs(la[i - 1] - lb[j - 1])
        if abs(D[i, j] - (D[i - 1, j - 1] + c)) < 1e-9:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif abs(D[i, j] - (D[i - 1, j] + skip_a)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def track_branches(points, x_key, y_key, gap=0.3):
    """
    Branch id for every root, per resistance.
    Returns {R: {(point_index, root_index): branch_id}}.

    Cells are visited row by row. Each cell is aligned to the state of its left
    neighbour (or the cell below if there is none). Branches missing at a cell
    are carried forward with their last frequency, so a root that drops out and
    returns keeps its id.
    """
    xs, ys, cell_of, where = _grid(points, x_key, y_key)
    all_r = sorted({r for pt in points for r in pt.result})
    out = {}
    for r in all_r: # for each resistance, track the branches
        lab, state_of, next_id = {}, {}, 0
        for j in np.arange(len(xs))[::-1]:
            for i in np.arange(len(ys))[::-1]: # for each cell in the grid, determine the branch ids of its roots                
                p = where.get((j, i)) # identify the point at this cell
                if p is None: # the cell is not in the sweep, so there are no roots to track
                    continue
                roots = np.asarray(points[p].result.get(r, []), float) # get the roots
                order = np.argsort(roots) # sort the roots
                sroots = roots[order] # sort the roots
                ids = [None] * len(roots) # initialize a list for the branch ids of the roots

                ref = None
                for nb in [(j-1, i-1), (j-1, i), (j-1, i+1),
                           (j, i-1), (j, i+1),
                           (j+1, i-1), (j+1, i), (j+1, i+1)]: # loop over the neighbors (left and below) to find a reference state
                    q = where.get(nb)
                    if q is not None and q in state_of: # if the neighbor exists and has a state, use it as a reference
                        if ref == None:
                            ref = q
                            continue
                        if len(state_of[q]) == len(roots): # if both neighbors have the same number of roots, they should belong to the same branches, so if they have different branch ids, the branch tracking is inconsistent.
                            if len(state_of[ref]) == len(state_of[q]):
                                # sort state_of[ref] by branch id:
                                state_of[ref] = sorted(state_of[ref], key=lambda s: s[1])
                                state_of[q] = sorted(state_of[q], key=lambda s: s[1])
                                if np.any(np.array([state_of[ref][k][1] for k in range(len(state_of[ref]))]) != np.array([state_of[q][k][1] for k in range(len(state_of[q]))])):
                                    for k in range(len(state_of[q])):
                                        state_of[ref][k] = (state_of[ref][k][0], min(state_of[ref][k][1], state_of[q][k][1]))
                                        state_of[q][k] = (state_of[q][k][0], state_of[ref][k][1])
                            ref = q
                            # If this happens, the states of the neighbors need to be adjusted, such that the lowest branch ids are used. INCORRECT LOGIC
                            # In the limit this should be correct, since the neighbors of neighbors will get arbitrarily close

                if ref is None: # if the neighbor has no state, assign new branch ids to the roots and create a new state
                    state = []
                    for k in range(len(roots)): # each root gets a unique new branch id
                        ids[k] = next_id; next_id += 1
                        state.append((roots[k], ids[k])) # the state of this grid point is the list of (frequency, branch_id) pairs
                else: 
                    """
                    If the neighbor does have a state (list of (frequency, branch_id) pairs), then compare the list of roots, 
                    If the number of roots is the same of - any neighbor with state -, copy the branch ids from the neighbor and update the corresponding frequencies.
                    #? What if two resonances cross? Can that occur?
                    Else, distinguish between (1) branch disappearance  (len(roots) = len(state_of[ref]) - 2) and (2) branch appearence (len(roots) = len(state_of[ref]) + 2).
                    (1)    
                        1. Check which roots are closest together in the neighboring cell, then update the branch ids of the roots in the current cell,
                            skipping the branch ids of the aforementioned roots.
                        2. Check each frequency of the current cell against the frequencies of the neighboring cell, 
                            assign the branch id of the closest frequency in the neighboring cell to the current frequency, and update the state of the current cell.
                    (2)
                        1. Check which roots are closest together in the current cell, then update the branch ids of the other roots in the current cell,
                            add new branch ids for the new roots, and update the state of the current cell.
                        2. Check each frequency of the neighboring cell against the frequencies of the current cell, 
                            assign the branch id of frequency in the neighboring cell to the closest frequency in the current cell to the current frequency,
                            add new branch ids for the remainder roots, and update the state of the current cell.
                    Assuming the the frequency lists are always ordered, the algorithm implementation can be done relatively easily.
                    """
                    # Neighbour state: list of (frequency, branch_id), sorted by frequency
                    st = sorted(state_of[ref], key=lambda s: s[0])
                    sf = np.array([s[0] for s in st])
                    ns, nr = len(st), len(roots)

                    # Same count: pairs in frequency order. Different count: the
                    # ordered matching drops the entries that fit worst.
                    # Each pair is (index in sorted roots, index in neighbour state).
                    pairs = _ordered_match(sroots, sf) if nr and ns else []

                    matched_state = {}                 # state index -> root index
                    for a_, b_ in pairs:
                        ids[order[a_]] = st[b_][1]     # inherit the branch id
                        matched_state[b_] = order[a_]

                    state = []
                    for b_, (f_, s_) in enumerate(st):
                        if b_ in matched_state:
                            state.append((roots[matched_state[b_]], s_))  # update frequency
                        else:
                            pass  # disappeared here: drop from state
                            # state.append((f_, s_))     # disappeared here: carry forward

                    for k in range(nr):
                        if ids[k] is None:             # appeared here: new branch
                            ids[k] = next_id; next_id += 1
                            state.append((roots[k], ids[k]))
                # sort state_of[ref] by branch id:
                state = sorted(state, key=lambda s: s[1])
                state_of[p] = state
                # for k, bid in enumerate(ids):
                #     lab[(p, k)] = bid # store the branch id for this root, defined by the point and the index, in this cell
                # print(f"Cell ({j}, {i}) at {xs[i]:g}, {ys[j]:g}: roots {roots} -> state {state}")
        # derive the output from the states of each point
        for p, state in state_of.items():
            for k in range(len(points[p].result[r])):
                for (f, bid) in state:
                    if f != points[p].result[r][k]:
                        continue
                    lab[(p, k)] = bid
                # print(f"Point ({points[p].params["C_P"]:.1e}, {points[p].params["L_P"]:.1e}), root {k}: frequency {f:g} -> branch id {bid}")
        out[r] = lab
    return out

def contour_points(cs):
    """
    Vertices of every contour line in a ContourSet.
    Returns a list of (level_MHz, x_array, y_array), one entry per line.
    x and y are in the same units as the axes (x_key and y_key values).
    """
    out = []
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            out.append((float(level), seg[:, 0], seg[:, 1]))
    return out

def plot_branches(points, labels, resistances=None, x_key="C_P", y_key="L_P",
                  cmap="viridis", contour_step=25):
    """
    One figure per branch. Each figure has one panel per resistance, with a
    coloured square per (x_key, y_key) cell and red lines at multiples of
    contour_step MHz. Branches are numbered by median frequency.
    """
    xs, ys, cell_of, _ = _grid(points, x_key, y_key)
    all_r = sorted(labels)
    if resistances is not None:
        wanted = np.atleast_1d(resistances)
        all_r = [r for r in all_r if np.any(np.isclose(r, wanted))]

    values = {}
    for r in all_r:
        for (p, k), b in labels[r].items():
            values.setdefault(b, []).append(points[p].result[r][k] * 1e-6)
    order = sorted(values, key=lambda b: np.median(values[b]))

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white", 1.0)

    figs = []
    css = []
    for number, b in enumerate(order, start=1):
        grids = {}
        for r in all_r:
            g = np.full((len(ys), len(xs)), np.nan)
            for (p, k), bid in labels[r].items():
                if bid == b:
                    j, i = cell_of[p]
                    g[j, i] = points[p].result[r][k] * 1e-6
            grids[r] = g

        finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()] or [np.array([])])
        if finite.size == 0:
            continue
        vmin, vmax = finite.min(), finite.max()

        fig, axes = plt.subplots(1, len(all_r), figsize=(5 * len(all_r), 4.5), squeeze=False)
        for ax, r in zip(axes[0], all_r):
            g = np.ma.masked_invalid(grids[r])
            mesh = ax.pcolormesh(xs, ys, g, shading="nearest", cmap=cm,
                                 vmin=vmin, vmax=vmax, edgecolors="k", linewidth=0.5)
            fig.colorbar(mesh, ax=ax, label="Frequency (MHz)")
            
            # Draw red contour lines at multiples of contour_step MHz
            if g.count() > 0 and len(xs) > 1 and len(ys) > 1:
                first = np.ceil(vmin / contour_step) * contour_step
                levels = np.arange(first, vmax + 1e-9, contour_step)
                if levels.size:
                    css.append(ax.contour(xs, ys, g, levels=levels, colors="red", linewidths=1, alpha=0.6))
                    # ax.clabel(css[-1], fmt="%g MHz", fontsize=8)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(f"R_eq = {r:g} ohm")
            # set logarithmic axes
            ax.set_xscale("log")
            ax.set_yscale("log")
        fig.suptitle(f"Branch {number} vs. {x_key} and {y_key}")
        fig.tight_layout()
        figs.append(fig)
    return figs, css

def plot_parametric_resonances(points, resistances=None, x_key="C_P", y_key="L_P",
                               cmap="viridis", contour_step=100.0):
    """
    One figure per root index n. Each figure has one panel per load resistance.
    Each panel is a grid of colored squares, one per (x_key, y_key) combination,
    colored by the nth root in MHz. Red lines mark multiples of contour_step (MHz).
    Missing combinations are left white.
    """
    xs = np.array(sorted({pt.params[x_key] for pt in points}))
    ys = np.array(sorted({pt.params[y_key] for pt in points}))

    all_r = sorted({r for pt in points for r in pt.result})
    if resistances is not None:
        wanted = np.atleast_1d(resistances)
        all_r = [r for r in all_r if np.any(np.isclose(r, wanted))]

    n_max = max((len(v) for pt in points for v in pt.result.values()), default=0)

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white", 1.0)

    figs = []
    for n in range(n_max):
        # One grid per resistance for this root index
        grids = {}
        for r in all_r:
            g = np.full((len(ys), len(xs)), np.nan)
            for pt in points:
                roots = pt.result.get(r)
                if roots is None or n >= len(roots):
                    continue
                i = int(np.searchsorted(xs, pt.params[x_key]))
                j = int(np.searchsorted(ys, pt.params[y_key]))
                g[j, i] = roots[n] * 1e-6
            grids[r] = g

        # One colour scale shared across the panels of this figure
        finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()] or [np.array([])])
        if finite.size == 0:
            continue
        vmin, vmax = finite.min(), finite.max()

        fig, axes = plt.subplots(1, len(all_r), figsize=(5 * len(all_r), 4.5), squeeze=False)
        for ax, r in zip(axes[0], all_r):
            g = np.ma.masked_invalid(grids[r])
            mesh = ax.pcolormesh(xs, ys, g, shading="nearest", cmap=cm,
                                 vmin=vmin, vmax=vmax, edgecolors="k", linewidth=0.5)
            fig.colorbar(mesh, ax=ax, label="Frequency (MHz)")

            # Red contour lines at multiples of contour_step MHz
            if np.isfinite(vmin) and g.count() > 0 and len(xs) > 1 and len(ys) > 1:
                first = np.ceil(vmin / contour_step) * contour_step
                levels = np.arange(first, vmax + 1e-9, contour_step)
                if levels.size:
                    cs = ax.contour(xs, ys, g, levels=levels, colors="red", linewidths=1.8)
                    ax.clabel(cs, fmt="%g MHz", fontsize=8)

            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(f"R_eq = {r:g} ohm, resonance {n + 1}")

        fig.suptitle(f"Resonance {n + 1} vs. {x_key} and {y_key}")
        fig.tight_layout()
        figs.append(fig)

    return figs


def plot_vna_traces(traces, title=None, mark_minima=True, log_x=False, ax=None):
    """
    S11 (dB) versus frequency (MHz) for one or more ``VnaTrace`` objects.

    Each trace is labelled by its resistance when known, otherwise by its name.
    With ``mark_minima``, the deepest point of each trace is marked and annotated.
    """
    if isinstance(traces, VnaTrace):
        traces = [traces]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for tr in traces:
        label = f"{tr.resistance_ohm:g} ohm" if tr.resistance_ohm is not None else tr.name
        line, = ax.plot(tr.freq_hz / 1e6, tr.s11_db, label=label)
        if mark_minima:
            f0, g0 = tr.minimum()
            ax.scatter([f0 / 1e6], [g0], color=line.get_color(), marker="x", s=40, zorder=3)
            ax.annotate(f"{f0 / 1e6:.2f} MHz, {g0:.2f} dB", (f0 / 1e6, g0),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("S11 (dB)")
    ax.set_title(title or "VNA S11")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_vna_vs_model(trace, resistance=None, cable=CableParams(), bridge=BridgeParams(),
                      title=None, ax=None, show_minima=True, min_depth_db=-3.0):
    """
    Overlay one VNA trace on the model's |Gamma| for the same load resistance.

    The model is evaluated on the VNA's own frequency points. ``resistance``
    defaults to the resistance read from the trace.
    """
    R = trace.resistance_ohm if resistance is None else resistance
    if R is None:
        raise ValueError("No resistance given and none found in the trace.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    f = trace.freq_hz
    meas_line, = ax.plot(f / 1e6, trace.s11_db, label="VNA measurement")
    model_db = reflection_spectrum_db(f, R, cable, bridge)
    model_line, = ax.plot(f / 1e6, model_db, linestyle="--", label=f"model, R = {R:g} ohm")

    if show_minima:
        mf, md = find_trace_minima(trace, min_depth_db=min_depth_db)
        ax.scatter(mf / 1e6, md, color=meas_line.get_color(), marker="o", s=30,
                   zorder=3, label="measured dips")
        model_res = resonance_sweep(f, [R], cable, bridge)
        pf = np.atleast_1d(model_res.resonance_frequencies[0])
        pg = np.atleast_1d(model_res.gamma_at_resonance_db[0])
        if min_depth_db is not None:
            keep = pg <= min_depth_db
            pf, pg = pf[keep], pg[keep]
        ax.scatter(pf / 1e6, pg, color=model_line.get_color(), marker="^", s=40,
                   zorder=3, label="model dips")

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("S11 / |Gamma| (dB)")
    ax.set_title(title or f"Measured vs model at {R:g} ohm")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_fit_resonances(result, measured_frequencies_hz=None, title=None):
    """
    Measured resonance frequency against load resistance, with the fitted model.

    ``result`` is a ``FitResult`` from ``fit_bridge``. The measured points are
    taken from the fit unless ``measured_frequencies_hz`` is given.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    f_meas = result.measured_frequencies_hz if measured_frequencies_hz is None \
        else np.asarray(measured_frequencies_hz, dtype=float)
    # Markers only: a resistance can have several dips (modes), so joining points would mislead
    R = result.resistances
    ax.semilogx(R, f_meas / 1e6, "o", label="measured")
    ax.semilogx(R, result.model_frequencies_hz / 1e6, "^", label="fitted model")
    ax.set_xlabel("Load resistance (ohm)")
    ax.set_ylabel("Resonance frequency (MHz)")
    ax.set_title(title or f"Bridge fit, RMS error {100 * result.rms_relative_error:.2f}%")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_fit_on_traces(traces, result, cable=None, title=None, ax=None):
    """
    Measured S11 (solid) with the fitted model's S11 (dashed) over the same frequencies.

    ``traces`` is a list of ``VnaTrace``; each model curve uses that trace's
    resistance and the fitted bridge in ``result`` (a ``FitResult``).
    """
    if isinstance(traces, VnaTrace):
        traces = [traces]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if cable is None:
        cable = getattr(result, "cable", CableParams(length=0.0))
    fitted_R = getattr(result, "fitted_resistances", None)
    colors = plt.cm.viridis(np.linspace(0, 1, len(traces)))
    for i, (c, tr) in enumerate(zip(colors, traces)):
        R = tr.resistance_ohm
        Rm = R if fitted_R is None else float(fitted_R[i])
        ax.plot(tr.freq_hz / 1e6, tr.s11_db, color=c, lw=1.2,
                label=f"{R:g} ohm measured")
        model_db = reflection_spectrum_db(tr.freq_hz, Rm, cable, result.bridge)
        ax.plot(tr.freq_hz / 1e6, model_db, color=c, lw=1.2, linestyle="--",
                label=f"{Rm:.0f} ohm fitted" if fitted_R is not None else f"{R:g} ohm fitted")

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("S11 / |Gamma| (dB)")
    ax.set_title(title or "Measured S11 (solid) and fitted model (dashed)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig
