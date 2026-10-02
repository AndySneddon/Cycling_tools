"""Plotly figures for the race planner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .course import Course
from .simulate import SimResult, fmt_time, gear_usage

GRADE_BINS = [
    (-99, -6, "< -6%", "#2a6fb0"),
    (-6, -3, "-6 to -3%", "#6fa8d8"),
    (-3, -1, "-3 to -1%", "#b7d4ea"),
    (-1, 1, "-1 to 1%", "#c9c9c9"),
    (1, 3, "1 to 3%", "#f2c57c"),
    (3, 6, "3 to 6%", "#e8803a"),
    (6, 99, "> 6%", "#c0392b"),
]
EVEN_COLOUR = "#7f8c8d"
OPT_COLOUR = "#1f77b4"


def _thin(n: int, target: int = 2500) -> slice:
    return slice(None, None, max(1, n // target))


def fig_elevation(course: Course, headwind_ms: np.ndarray | None = None) -> go.Figure:
    """Elevation profile shaded by gradient, optional headwind overlay (km/h, +ve = headwind)."""
    sl = _thin(course.n_seg)
    x = course.seg_mid_m[sl] / 1000.0
    y = 0.5 * (course.elev_m[:-1] + course.elev_m[1:])[sl]
    g = course.grade[sl] * 100.0
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    base = float(y.min()) - 0.1 * max(float(np.ptp(y)), 10.0)
    for lo, hi, label, col in GRADE_BINS:
        m = (g >= lo) & (g < hi)
        if not m.any():
            continue
        yy = np.where(m, y, np.nan)
        # include neighbours so adjacent bins join without gaps
        edge = m | np.roll(m, 1) | np.roll(m, -1)
        yy = np.where(edge, y, np.nan)
        fig.add_trace(go.Scatter(x=x, y=yy, mode="lines", name=label, line=dict(width=0.5, color=col),
                                 fill="tozeroy", fillcolor=col, connectgaps=False,
                                 hovertemplate="%{x:.1f} km<br>%{y:.0f} m<extra>" + label + "</extra>"),
                      secondary_y=False)
    if headwind_ms is not None:
        fig.add_trace(go.Scatter(x=x, y=(headwind_ms * 3.6)[sl], mode="lines", name="Headwind (km/h)",
                                 line=dict(color="#222", width=1.2, dash="dot"),
                                 hovertemplate="%{x:.1f} km<br>%{y:.1f} km/h<extra>headwind</extra>"),
                      secondary_y=True)
        fig.update_yaxes(title_text="Headwind (km/h, +ve = into wind)", secondary_y=True, showgrid=False,
                         zeroline=True)
    fig.update_yaxes(title_text="Elevation (m)", range=[base, float(y.max()) + 0.1 * max(float(np.ptp(y)), 10.0)],
                     secondary_y=False)
    fig.update_xaxes(title_text="Distance (km)")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3),
                      title="Course profile (shaded by gradient)", hovermode="x unified")
    return fig


def fig_map(course: Course, result: SimResult | None = None, colour_by: str = "speed") -> go.Figure:
    """Route map on OpenStreetMap tiles (no token), coloured by speed, power or gradient."""
    import plotly.express as px

    sl = _thin(course.n_seg, 1500)
    lat = 0.5 * (course.lat[:-1] + course.lat[1:])[sl]
    lon = 0.5 * (course.lon[:-1] + course.lon[1:])[sl]
    if result is not None and colour_by == "speed":
        z, label = result.v_seg[sl] * 3.6, "Speed (km/h)"
    elif result is not None and colour_by == "power":
        z, label = result.power_applied[sl], "Power (W)"
    else:
        z, label = course.grade[sl] * 100.0, "Gradient (%)"
    df = pd.DataFrame({"lat": lat, "lon": lon, "z": z, "km": course.seg_mid_m[sl] / 1000.0})
    zoom = 11 if course.length_m < 60e3 else 10
    cscale = "RdBu_r" if colour_by == "grade" or result is None else "Viridis"
    fig = px.scatter_map(df, lat="lat", lon="lon", color="z", color_continuous_scale=cscale,
                         hover_data={"km": ":.1f", "z": ":.1f", "lat": False, "lon": False},
                         labels={"z": label}, zoom=zoom, map_style="open-street-map", height=450)
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=label))
    fig.update_layout(map=dict(center=dict(lat=float(course.lat.mean()), lon=float(course.lon.mean()))))
    return fig


def _smooth_per_dist(values: np.ndarray, course: Course, window_m: float) -> np.ndarray:
    k = max(1, int(window_m / float(np.mean(course.ds))))
    if k == 1:
        return values
    return np.convolve(np.pad(values, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def fig_speed_power(even: SimResult, opt: SimResult | None = None, smooth_m: float = 100.0) -> go.Figure:
    """Speed and power vs distance for even (and optionally optimised) pacing."""
    c = even.course
    sl = _thin(c.n_seg)
    x = c.seg_mid_m[sl] / 1000.0
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.5, 0.5, 0.25],
                        subplot_titles=("Power (W)", "Speed (km/h)", "Elevation (m)"))
    for res, name, col in ((even, "Even", EVEN_COLOUR), (opt, "Optimised", OPT_COLOUR)):
        if res is None:
            continue
        fig.add_trace(go.Scatter(x=x, y=_smooth_per_dist(res.power_applied, c, smooth_m)[sl], name=name,
                                 line=dict(color=col, width=1.6), legendgroup=name,
                                 hovertemplate="%{x:.1f} km %{y:.0f} W<extra>" + name + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=(res.v_seg * 3.6)[sl], name=name, showlegend=False,
                                 line=dict(color=col, width=1.4), legendgroup=name,
                                 hovertemplate="%{x:.1f} km %{y:.1f} km/h<extra>" + name + "</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=0.5 * (c.elev_m[:-1] + c.elev_m[1:])[sl], fill="tozeroy", name="Elevation",
                             line=dict(color="#95a5a6", width=1), showlegend=False), row=3, col=1)
    fig.update_xaxes(title_text="Distance (km)", row=3, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    return fig


def fig_np_curve(curve: pd.DataFrame, current_np: float | None = None) -> go.Figure:
    """Predicted finish time vs normalised power (even pacing)."""
    fig = go.Figure(go.Scatter(
        x=curve["np_w"], y=curve["time_s"] / 60.0, mode="lines+markers", line=dict(color=OPT_COLOUR),
        customdata=np.column_stack([curve["time"], curve["avg_speed_kmh"], curve["s_saved_per_5w"]]),
        hovertemplate="NP %{x:.0f} W<br>%{customdata[0]} (%{customdata[1]:.1f} km/h)<br>"
                      "+5 W saves %{customdata[2]:.0f} s<extra></extra>"))
    if current_np is not None:
        t = float(np.interp(current_np, curve["np_w"], curve["time_s"]))
        fig.add_trace(go.Scatter(x=[current_np], y=[t / 60.0], mode="markers", name="Selected",
                                 marker=dict(size=13, color="#c0392b"), showlegend=False,
                                 hovertemplate=f"NP {current_np:.0f} W<br>{fmt_time(t)}<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Finish time vs normalised power",
                      xaxis_title="Normalised power (W)", yaxis_title="Predicted time (min)")
    return fig


def fig_gear_heatmap(result: SimResult, chainrings, cassette=None) -> go.Figure:
    """% of pedalling time in each sprocket for each candidate chainring."""
    usage = gear_usage(result, chainrings, cassette)
    z = usage.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(s) for s in usage.columns], y=[f"{r}T" for r in usage.index], colorscale="Blues",
        text=np.where(z >= 0.5, np.round(z, 0).astype(int).astype(str), ""), texttemplate="%{text}",
        hovertemplate="%{y} / %{x}T: %{z:.1f}% of pedalling time<extra></extra>",
        colorbar=dict(title="% time")))
    fig.update_layout(height=60 + 34 * len(usage), margin=dict(l=10, r=10, t=30, b=10),
                      title="Gear usage on this course (sprocket at target cadence)",
                      xaxis_title="Sprocket (teeth)", yaxis_title="Chainring", yaxis=dict(autorange="reversed"))
    return fig


def fig_split_table(splits: pd.DataFrame) -> go.Figure:
    cols = ["split", "split_time", "cum_time", "speed_kmh", "power_w", "net_elev_m", "headwind_ms"]
    heads = ["Split", "Time", "Cumulative", "km/h", "Power (W)", "Net elev (m)", "Headwind (m/s)"]
    cells = [splits[c].round(1).tolist() if splits[c].dtype.kind == "f" else splits[c].tolist() for c in cols]
    return go.Figure(go.Table(header=dict(values=heads, fill_color="#34495e", font=dict(color="white")),
                              cells=dict(values=cells, align="left")))


def fig_pacing_blocks(opt, show_even: bool = True) -> go.Figure:
    """Power target per block as % of NP, with gradient context."""
    t = opt.block_table()
    fig = go.Figure(go.Bar(x=(t["start_km"] + t["end_km"]) / 2, y=t["pct_of_np"], width=(t["end_km"] - t["start_km"]) * 0.95,
                           marker_color=np.where(t["pct_of_np"] >= 100, "#c0392b", "#2a6fb0"),
                           customdata=np.column_stack([t["target_power_w"], t["grade_pct"], t["headwind_ms"]]),
                           hovertemplate="%{x:.1f} km<br>%{y:.0f}% NP (%{customdata[0]:.0f} W)<br>"
                                         "grade %{customdata[1]:.1f}%, headwind %{customdata[2]:.1f} m/s<extra></extra>"))
    fig.add_hline(y=100, line_dash="dot", line_color="#555")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Optimised power targets (% of NP)",
                      xaxis_title="Distance (km)", yaxis_title="% of NP")
    return fig
