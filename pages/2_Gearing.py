"""Best chainring analyser."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from cycling_tools.branding import apply_branding

from cycling_tools import gearing as g
from cycling_tools import viz_gearing as vz
from cycling_tools.fit_io import load_fit
from cycling_tools.profile import RiderProfile

st.set_page_config(page_title="Gearing", layout="wide")
apply_branding()
st.title("Best chainring analyser")
st.caption("Assumes you keep your observed speed and cadence preference when swapping chainrings. "
           "The nearest real sprocket is used, so 'cadence error' shows the cost of gear-step quantisation.")

profile = RiderProfile.load()
FIT_DIR = ROOT / "Fit_files"


@st.cache_data(show_spinner=False)
def _load(key: str, data: bytes | None):
    return load_fit(FIT_DIR / key) if data is None else load_fit(data, name=key)


with st.sidebar:
    st.header("Rides")
    avail = sorted(p.name for p in FIT_DIR.glob("*") if p.suffix.lower() == ".fit")
    chosen = st.multiselect("FIT files in Fit_files/", avail, default=avail[:1])
    ups = st.file_uploader("...or upload FIT files", type=["fit"], accept_multiple_files=True)

rides = []
for n in chosen:
    try:
        rides.append(_load(n, None))
    except Exception as e:  # noqa: BLE001
        st.error(f"{n}: {e}")
for u in ups or []:
    try:
        rides.append(_load(u.name, u.getvalue()))
    except Exception as e:  # noqa: BLE001
        st.error(f"{u.name}: {e}")
if not rides:
    st.info("Pick or upload at least one FIT file.")
    st.stop()

laps = {}
with st.sidebar:
    st.header("Laps")
    for r in rides:
        nlap = int(r.df["lap"].max())
        if nlap > 1:
            sel = st.multiselect(f"{r.name}", list(range(1, nlap + 1)), default=list(range(1, nlap + 1)),
                                 key=f"laps_{r.name}")
            laps[r.name] = sel

    st.header("Cassette")
    preset = st.selectbox("Preset", ["Custom"] + list(g.CASSETTES),
                          index=list(g.CASSETTES).index("Ultegra 11-30 (12sp)") + 1)
    default_txt = ",".join(map(str, profile.cassette if preset == "Custom" else g.CASSETTES[preset]))
    cas_txt = st.text_input("Sprockets", default_txt, key=f"cas_{preset}")
    circ = st.number_input("Tyre circumference (m)", 1.8, 2.4, float(profile.tyre_circumference_m), 0.005)

    st.header("Drivetrain")
    mode = st.radio("Mode", ["1x only", "2x only", "Both compared"], index=2)
    setups = []
    if mode != "2x only":
        c1, c2 = st.columns(2)
        lo = c1.number_input("1x min ring", 34, 70, 50)
        hi = c2.number_input("1x max ring", 36, 72, 64)
        step = st.selectbox("1x step", [1, 2, 4], index=1)
        setups += [g.Setup((r,)) for r in range(int(lo), int(hi) + 1, int(step))]
    if mode != "1x only":
        pairs = st.multiselect("2x pairs (big/small)", g.DEFAULT_2X_PAIRS, default=["56/42", "54/40", "52/36"])
        custom = st.text_input("Custom pairs (comma separated, e.g. 55/41, 58/42)", "")
        names = list(pairs) + [x.strip() for x in custom.split(",") if x.strip()]
        for nme in dict.fromkeys(names):
            try:
                setups.append(g.parse_setup(nme))
            except ValueError as e:
                st.error(f"{nme}: {e}")

    st.header("Filters")
    terrain = st.selectbox("Terrain", ["all", "climb", "flat", "descent"])
    min_power = st.slider("Min power (W)", 0, 300, 50)
    pr = st.slider("Power range (W)", 0, 1000, (0, 1000))
    min_cad = st.slider("Min cadence (rpm)", 0, 90, 40)

try:
    cassette = g.parse_cassette(cas_txt)
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.expander("Score weights"):
    w_ends = st.slider("Penalty per % time on outer 2 sprockets", 0.0, 2.0, g.DEFAULT_WEIGHTS["w_ends"], 0.05)
    w_oor = st.slider("Penalty per % time out of range", 0.0, 5.0, g.DEFAULT_WEIGHTS["w_oor"], 0.1)
    st.markdown("**2x only.** The aero cost of a front derailleur + second ring is an *assumption* - change it.")
    d_cda = st.slider("Aero penalty for 2x (CdA, m2)", 0.0, 0.01, g.DEFAULT_SETUP_WEIGHTS["aero_delta_cda"], 0.0005,
                      format="%.4f")
    w_aero = st.slider("Score points per watt of aero penalty", 0.0, 10.0, g.DEFAULT_SETUP_WEIGHTS["w_aero"], 0.5)
    w_cross = st.slider("Penalty per % time cross-chained", 0.0, 2.0, g.DEFAULT_SETUP_WEIGHTS["w_cross"], 0.05)
    w_shift = st.slider("Penalty per front shift per hour", 0.0, 0.5, g.DEFAULT_SETUP_WEIGHTS["w_shift"], 0.01)
    n_cross = st.slider("Sprockets counted as cross-chained at each end", 0, 5, g.DEFAULT_SETUP_WEIGHTS["n_cross"])
wts = {"w_ends": w_ends, "w_oor": w_oor, "aero_delta_cda": d_cda, "w_aero": w_aero, "w_cross": w_cross,
       "w_shift": w_shift, "n_cross": n_cross}

df = g.prepare_gearing_samples(rides, min_power=min_power, min_cadence=min_cad, laps=laps, terrain=terrain,
                               power_range=(pr[0], None if pr[1] >= 1000 else pr[1]))
if len(df) < 60:
    st.warning("Too few pedalling samples with these filters.")
    st.stop()
if not setups:
    st.warning("Choose at least one chainring setup.")
    st.stop()

res = g.evaluate_setups(df["speed"], df["cadence"], df["weight_s"], setups, cassette, circ, wts)
ranked = res.sort_values("score", ascending=False).reset_index(drop=True)
best = ranked.iloc[0]
cm = g.fit_cadence_model(df)

st.success(f"Recommended: **{best.setup}** with {cassette[0]}-{cassette[-1]} "
           f"(score {best.score:.1f}; {best.mid4_pct:.0f}% in middle 4, {best.ends_pct:.0f}% on outer ends, "
           f"{best.out_of_range_pct:.1f}% out of range, {best.cross_chain_pct:.1f}% cross-chained). "
           f"Based on {len(df)/3600:.1f} h of pedalling.")
one, two = ranked[ranked.n_rings == 1], ranked[ranked.n_rings == 2]
if len(one) and len(two):
    b1, b2 = one.iloc[0], two.iloc[0]
    d = b2.score - b1.score
    st.info(f"**1x vs 2x:** best 1x is {b1.setup} (score {b1.score:.1f}); best 2x is {b2.setup} (score {b2.score:.1f}, "
            f"includes {b2.aero_penalty_w:.1f} W aero penalty assumption and {b2.front_shifts_per_hour:.0f} front shifts/h). "
            f"{'2x' if d > 0 else '1x'} wins by {abs(d):.1f} points. Gear range: {b1.gear_range_pct:.0f}% vs {b2.gear_range_pct:.0f}%.")
st.caption("Assumptions: you keep your observed speed and cadence when changing setup; the 2x aero penalty "
           f"({d_cda:.4f} m2 CdA) is a guess you can change in 'Score weights'.")
m1, m2, m3 = st.columns(3)
m1.metric("Cadence at median power", f"{cm['cadence_flat']:.0f} rpm", f"median {cm['ref_power']:.0f} W", delta_color="off")
m2.metric("Cadence slope", f"{cm['cadence_per_100w']:+.1f} rpm / 100 W")
m3.metric("Median cadence", f"{df['cadence'].median():.0f} rpm")

st.plotly_chart(vz.ranking_fig(ranked), use_container_width=True)
st.dataframe(ranked.round(1), hide_index=True, use_container_width=True)

st.subheader("Time by gear")
ones = [x for x in setups if not x.is_2x]
if ones:
    st.caption("1x overview: % of time each sprocket would be used per chainring")
    st.plotly_chart(vz.heatmap_fig(g.usage_matrix(df["speed"], df["cadence"], df["weight_s"],
                                                  [x.chainrings[0] for x in ones], cassette, circ)),
                    use_container_width=True)
labels = [x.label for x in setups]
sel = st.selectbox("Setup to inspect", labels, index=labels.index(best.setup))
chosen_setup = setups[labels.index(sel)]
if chosen_setup.is_2x:
    st.caption("Gear usage per ring/sprocket combo (dotted red = cross-chained)")
    st.plotly_chart(vz.heatmap_fig(g.usage_matrix_setup(df["speed"], df["cadence"], df["weight_s"], chosen_setup,
                                                        cassette, circ, wts), chosen_setup, n_cross),
                    use_container_width=True)
    seq = g.gear_sequence(df["speed"], df["cadence"], df["weight_s"], chosen_setup, cassette, circ, wts)
    st.caption("Front ring in use over the (concatenated) pedalling time")
    st.plotly_chart(vz.ring_strip_fig(seq, chosen_setup), use_container_width=True)
a, b = st.columns(2)
a.plotly_chart(vz.speed_cadence_fig(df, chosen_setup, cassette, circ), use_container_width=True)
b.plotly_chart(vz.cadence_hist_fig(df, chosen_setup, cassette, circ, wts), use_container_width=True)
st.subheader("Power by gear")
st.plotly_chart(vz.power_by_gear_fig(df, chosen_setup, cassette, circ, wts), use_container_width=True)

if st.button("Save setup / cassette / cadence model to rider profile"):
    bs = setups[labels.index(best.setup)]
    profile.chainring = int(max(bs.chainrings))
    profile.cassette = cassette
    profile.tyre_circumference_m = float(circ)
    profile.cadence_flat = round(cm["cadence_flat"], 1)
    profile.cadence_per_100w = round(cm["cadence_per_100w"], 2)
    profile.notes["cadence_ref_power"] = round(cm["ref_power"])
    profile.notes["setup"] = bs.label
    profile.save()
    st.success(f"Saved {bs.label} to rider profile.")
