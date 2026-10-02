"""Plotly figures for the CdA estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .cda import REASONS, REASON_LABELS, CdAConfig, CdAResult, virtual_elevation

REASON_COLOURS = {
    "gap": "#9e9e9e", "low_speed": "#795548", "coasting": "#42a5f5", "braking": "#e53935",
    "cornering": "#fb8c00", "steep_grade": "#8e24aa", "high_accel": "#fdd835", "low_airspeed": "#26a69a",
}


def _selected(res: CdAResult) -> pd.DataFrame:
    return res.series[res.series["selected"]]


def _runs(mask: np.ndarray):
    """Yield (start, end) index pairs of True runs."""
    m = np.asarray(mask, bool)
    if not m.any():
        return
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        yield a, b - 1


def series_figure(res: CdAResult, x_axis: str = "time", show_masks: bool = True) -> go.Figure:
    s = _selected(res).reset_index(drop=True)
    if x_axis == "distance":
        x = s["dist_km"].to_numpy()
        xt = "Distance (km)"
    else:
        x = (s["t_s"] - s["t_s"].iloc[0]).to_numpy() / 60.0
        xt = "Time (min)"
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=("Power (W)", "Speed (km/h)", "Headwind (m/s)", "Rolling CdA (m²)"))
    fig.add_trace(go.Scatter(x=x, y=s["power"], name="Power", line=dict(color="#d81b60", width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=s["speed"] * 3.6, name="Speed", line=dict(color="#1e88e5", width=1)), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=s["headwind"], name="Headwind", line=dict(color="#43a047", width=1)), 3, 1)
    fig.add_hline(y=0, row=3, col=1, line=dict(color="grey", width=0.5))
    fig.add_trace(go.Scatter(x=x, y=s["rolling_cda"], name="Rolling CdA", line=dict(color="#6d4c41", width=1.5)), 4, 1)
    fig.add_hline(y=res.cda, row=4, col=1, line=dict(color="black", dash="dash", width=1),
                  annotation_text=f"overall {res.cda:.3f}")
    if show_masks:
        taken = np.zeros(len(s), bool)
        for k in REASONS:
            mk = s["mask_" + k].to_numpy() & ~taken
            taken |= mk
            for a, b in _runs(mk):
                fig.add_vrect(x0=x[a], x1=x[min(b + 1, len(x) - 1)], fillcolor=REASON_COLOURS[k], opacity=0.22,
                              line_width=0, layer="below")
        for k in REASONS:  # legend proxies
            if s["mask_" + k].any():
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=REASON_LABELS[k],
                                         marker=dict(size=10, color=REASON_COLOURS[k], symbol="square")), 1, 1)
    fig.update_xaxes(title_text=xt, row=4, col=1)
    fig.update_layout(height=760, margin=dict(l=50, r=20, t=40, b=40), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.08))
    return fig


def map_figure(res: CdAResult) -> go.Figure:
    s = _selected(res).dropna(subset=["lat", "lon"])
    s = s.iloc[::3]
    if s.empty:
        return go.Figure().update_layout(title="No GPS data")
    vals = s["rolling_cda"]
    lo, hi = (np.nanpercentile(vals, [5, 95]) if vals.notna().any() else (0.2, 0.3))
    fig = px.scatter_map(s, lat="lat", lon="lon", color="rolling_cda", color_continuous_scale="Turbo",
                         range_color=(lo, hi), zoom=11, height=520,
                         hover_data={"speed": ":.1f", "power": ":.0f", "lat": False, "lon": False})
    fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_colorbar=dict(title="CdA"))
    return fig


def scatter_figure(res: CdAResult, by: str = "speed") -> go.Figure:
    s = res.series
    m = (s["valid"] & s["selected"]).to_numpy()
    d = s[m]
    xcol = d["speed"] * 3.6 if by == "speed" else d["headwind"]
    lab = "Speed (km/h)" if by == "speed" else "Headwind (m/s)"
    ic = d["implied_cda"].to_numpy()
    ok = np.isfinite(ic) & (ic > -0.2) & (ic < 0.8)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=xcol.to_numpy()[ok], y=ic[ok], mode="markers", name="Implied CdA (1 s)",
                               marker=dict(size=3, opacity=0.25, color="#90a4ae")))
    xs = xcol.to_numpy()[ok]
    ys = ic[ok]
    if len(xs) > 50:
        edges = np.quantile(xs, np.linspace(0, 1, 13))
        idx = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, 11)
        bx = [np.median(xs[idx == i]) for i in range(12) if (idx == i).sum() > 5]
        by_ = [np.median(ys[idx == i]) for i in range(12) if (idx == i).sum() > 5]
        fig.add_trace(go.Scatter(x=bx, y=by_, mode="lines+markers", name="Binned median",
                                 line=dict(color="#d81b60", width=3)))
    fig.add_hline(y=res.cda, line=dict(dash="dash", color="black"), annotation_text=f"fit {res.cda:.3f}")
    fig.update_layout(height=420, xaxis_title=lab, yaxis_title="Implied CdA (m²)", yaxis_range=[-0.1, 0.7],
                      margin=dict(l=50, r=20, t=30, b=40))
    return fig


def histogram_figure(res: CdAResult) -> go.Figure:
    s = res.series
    d = s[(s["valid"] & s["selected"])]["rolling_cda"].dropna()
    fig = px.histogram(d, nbins=40, height=360, labels={"value": "Rolling CdA (m²)"})
    fig.add_vline(x=res.cda, line=dict(dash="dash", color="black"))
    lo, hi = res.cda_ci
    if np.isfinite(lo):
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#d81b60", opacity=0.15, line_width=0)
    fig.update_layout(showlegend=False, margin=dict(l=50, r=20, t=30, b=40), yaxis_title="Seconds")
    return fig


def laps_figure(res: CdAResult) -> go.Figure:
    d = res.laps.copy()
    d["label"] = d["lap"].astype(str)
    colours = np.where(d["selected"], "#1e88e5", "#b0bec5")
    fig = go.Figure(go.Bar(x=d["label"], y=d["cda"], marker_color=colours,
                           customdata=np.c_[d["avg_power"], d["valid_pct"]],
                           hovertemplate="Lap %{x}<br>CdA %{y:.3f}<br>%{customdata[0]:.0f} W, %{customdata[1]:.0f}% valid<extra></extra>"))
    fig.add_hline(y=res.cda, line=dict(dash="dash", color="black"))
    fig.update_layout(height=340, xaxis_title="Lap", yaxis_title="CdA (m²)", margin=dict(l=50, r=20, t=30, b=40))
    return fig


def energy_figure(res: CdAResult, selection: bool = False) -> go.Figure:
    e = res.energy_selection if selection else res.energy
    if not e:
        return go.Figure()
    names = ["Aero", "Rolling", "Climb", "Accel", "Unexplained"]
    vals = [e["aero"], e["rolling"], e["climb"], e["accel"], e["unexplained"]]
    fig = go.Figure(go.Waterfall(
        x=names + ["Wheel power"], y=vals + [0], measure=["relative"] * 5 + ["total"],
        text=[f"{v:.0f} W" for v in vals] + [f"{e['measured_wheel']:.0f} W"], textposition="outside",
    ))
    fig.update_layout(height=360, yaxis_title="Average wheel power (W)", margin=dict(l=50, r=20, t=30, b=40))
    return fig


def ve_figure(res: CdAResult, cda: float, crr: float, cfg: CdAConfig, start: int | None = None,
              end: int | None = None, x_axis: str = "distance") -> go.Figure:
    s = _selected(res)
    ve = virtual_elevation(s.reset_index(drop=True), cda, crr, cfg, start, end)
    x = ve["dist_km"] if x_axis == "distance" else (ve["t_s"] - ve["t_s"].iloc[0]) / 60
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ve["alt"], name="Actual altitude", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=x, y=ve["ve"], name=f"Virtual elevation (CdA {cda:.3f})", line=dict(color="#d81b60")))
    fig.update_layout(height=400, xaxis_title="Distance (km)" if x_axis == "distance" else "Time (min)",
                      yaxis_title="Elevation (m)", margin=dict(l=50, r=20, t=30, b=40),
                      legend=dict(orientation="h", y=1.1))
    return fig


def wind_scan_figure(scan: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=scan["wind_scale"], y=scan["cda_head"], name="CdA headwind"), secondary_y=False)
    fig.add_trace(go.Scatter(x=scan["wind_scale"], y=scan["cda_tail"], name="CdA tailwind"), secondary_y=False)
    fig.add_trace(go.Scatter(x=scan["wind_scale"], y=scan["rms_w"], name="Residual RMS (W)",
                             line=dict(dash="dot", color="grey")), secondary_y=True)
    fig.update_layout(height=360, xaxis_title="Wind scale", margin=dict(l=50, r=50, t=30, b=40))
    fig.update_yaxes(title_text="CdA (m²)", secondary_y=False, range=[0, 0.8])
    fig.update_yaxes(title_text="RMS (W)", secondary_y=True)
    return fig
