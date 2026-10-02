"""Plotly figures for the gearing tool (1x and 2x)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .gearing import (DEFAULT_CIRCUMFERENCE_M, Setup, _assign_gears, _setup_weights, cadence_for,
                      cross_chain_mask, parse_setup)


def _as_setup(x) -> Setup:
    return x if isinstance(x, Setup) else (Setup((int(x),)) if np.ndim(x) == 0 and not isinstance(x, str)
                                           else parse_setup(x))


def ranking_fig(res: pd.DataFrame) -> go.Figure:
    """Stacked time-share bars + score line. Uses the 'setup' column if present, else 'chainring'."""
    r = res.copy()
    if "setup" in r:
        label = r["setup"].astype(str)
    else:
        r = r.sort_values("chainring")
        label = r["chainring"].astype(str)
    other = (100 - r["mid4_pct"] - r["ends_pct"] - r["out_of_range_pct"]
             - (r["cross_chain_pct"] if "cross_chain_pct" in r else 0)).clip(lower=0)
    fig = go.Figure()
    bars = [("Middle 4 sprockets", r["mid4_pct"], "#2a9d8f"), ("Other sprockets", other, "#b8c4cc"),
            ("Outer 2 each end", r["ends_pct"], "#e9c46a")]
    if "cross_chain_pct" in r:
        bars.append(("Cross-chained", r["cross_chain_pct"], "#9b5de5"))
    bars.append(("Out of range", r["out_of_range_pct"], "#e76f51"))
    for name, y, col in bars:
        fig.add_bar(x=label, y=y, name=name, marker_color=col)
    fig.add_scatter(x=label, y=r["score"], name="Score", mode="lines+markers", yaxis="y2",
                    line=dict(color="#264653", width=3))
    fig.update_layout(barmode="stack", xaxis_title="Setup (chainring[s] in T)", yaxis_title="% of pedalling time",
                      xaxis=dict(type="category"),
                      yaxis2=dict(title="Score", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=-0.3), margin=dict(t=30), height=440)
    return fig


def heatmap_fig(mat: pd.DataFrame, setup: Setup | None = None, n_cross: int = 3) -> go.Figure:
    """% time per ring x sprocket. If a 2x setup is given, cross-chained cells get a red outline."""
    fig = go.Figure(go.Heatmap(z=mat.to_numpy(), x=[str(c) for c in mat.columns], y=[str(i) for i in mat.index],
                               colorscale="Viridis", colorbar=dict(title="% time"),
                               hovertemplate="ring %{y}T / %{x}T: %{z:.1f}%<extra></extra>"))
    if setup is not None and setup.is_2x:
        mask = cross_chain_mask(setup, mat.shape[1], n_cross)
        for i, j in zip(*np.nonzero(mask)):
            fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                          line=dict(color="#e76f51", width=2, dash="dot"))
        fig.add_annotation(text="dotted red = cross-chained", xref="paper", yref="paper", x=1, y=-0.25,
                           showarrow=False, font=dict(size=11))
    fig.update_layout(xaxis_title="Sprocket (T)", yaxis_title="Chainring (T)", xaxis=dict(type="category"),
                      yaxis=dict(type="category"), height=max(250, 60 * len(mat) + 160), margin=dict(t=30))
    return fig


def speed_cadence_fig(df: pd.DataFrame, setup, cassette: list[int], circ: float = DEFAULT_CIRCUMFERENCE_M) -> go.Figure:
    setup = _as_setup(setup)
    fig = go.Figure(go.Histogram2d(x=df["speed"] * 3.6, y=df["cadence"], nbinsx=60, nbinsy=50,
                                   colorscale="Blues", colorbar=dict(title="samples")))
    v = np.linspace(max(df["speed"].min(), 1) * 3.6, df["speed"].max() * 3.6, 50)
    cols = ["rgba(231,111,81,0.8)", "rgba(155,93,229,0.7)"]
    for k, ring in enumerate(setup.chainrings):
        for s in sorted(cassette):
            fig.add_scatter(x=v, y=cadence_for(v / 3.6, ring, s, circ), mode="lines",
                            line=dict(width=1, color=cols[k]), name=f"{ring}x{s}",
                            hovertemplate=f"{ring}x{s}<extra></extra>", showlegend=False)
    fig.update_layout(xaxis_title="Speed (km/h)", yaxis_title="Cadence (rpm)", height=450,
                      yaxis=dict(range=[30, 140]), margin=dict(t=40), title=f"Iso-gear lines for {setup.label}")
    return fig


def cadence_hist_fig(df: pd.DataFrame, setup, cassette: list[int], circ: float = DEFAULT_CIRCUMFERENCE_M,
                     weights=None) -> go.Figure:
    setup = _as_setup(setup)
    a = _assign_gears(df["speed"].to_numpy(), df["cadence"].to_numpy(), setup, cassette, circ, _setup_weights(weights))
    fig = go.Figure()
    fig.add_histogram(x=df["cadence"], name="Observed", xbins=dict(size=2), opacity=0.6, marker_color="#264653")
    fig.add_histogram(x=a["cadence"][~a["oor"]], name=f"Achieved with {setup.label}", xbins=dict(size=2),
                      opacity=0.6, marker_color="#e9c46a")
    fig.update_layout(barmode="overlay", xaxis_title="Cadence (rpm)", yaxis_title="Samples (s)",
                      height=380, margin=dict(t=30), legend=dict(orientation="h", y=-0.25))
    return fig


def power_by_gear_fig(df: pd.DataFrame, setup, cassette: list[int], circ: float = DEFAULT_CIRCUMFERENCE_M,
                      weights=None) -> go.Figure:
    setup = _as_setup(setup)
    cas = np.asarray(sorted(cassette))
    a = _assign_gears(df["speed"].to_numpy(), df["cadence"].to_numpy(), setup, cas, circ, _setup_weights(weights))
    ring = np.asarray(setup.chainrings)[a["ring_idx"]]
    d = pd.DataFrame({"ring": ring, "spr": cas[a["sprocket_idx"]], "power": df["power"].to_numpy()})[~a["oor"]]
    d["ratio"] = d["ring"] / d["spr"]
    d["gear"] = d["ring"].astype(str) + "x" + d["spr"].astype(str)
    order = d.groupby("gear")["ratio"].first().sort_values().index
    fig = go.Figure()
    for gname in order:
        fig.add_box(y=d.loc[d["gear"] == gname, "power"], name=gname, boxpoints=False, marker_color="#2a9d8f")
    fig.update_layout(xaxis_title=f"Gear ({setup.label}), easy to hard", yaxis_title="Power (W)", height=380,
                      showlegend=False, margin=dict(t=30))
    return fig


def ring_strip_fig(seq: pd.DataFrame, setup: Setup, max_points: int = 4000) -> go.Figure:
    """Which front ring is in use over time (2x), cross-chained stretches shaded red. seq from gear_sequence()."""
    step = max(1, len(seq) // max_points)
    d = seq.iloc[::step]
    t = d["t_s"] / 60.0
    fig = go.Figure()
    fig.add_scatter(x=t, y=d["ring"].astype(str), mode="lines", line=dict(shape="hv", color="#264653"),
                    name="Ring in use")
    cx = d[d["crossed"]]
    fig.add_scatter(x=cx["t_s"] / 60.0, y=cx["ring"].astype(str), mode="markers",
                    marker=dict(color="#e76f51", size=5), name="Cross-chained")
    fig.update_layout(xaxis_title="Pedalling time (min)", yaxis_title="Front ring (T)",
                      yaxis=dict(type="category", categoryorder="array",
                                 categoryarray=[str(r) for r in sorted(setup.chainrings)]),
                      height=300, margin=dict(t=30), legend=dict(orientation="h", y=-0.35))
    return fig
