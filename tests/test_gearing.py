import glob
from pathlib import Path

import numpy as np
import pytest

from cycling_tools import gearing as g

CAS = g.CASSETTES["Ultegra 11-30 (12sp)"]


def test_maths_roundtrip():
    v = 10.0
    cad = g.cadence_for(v, 56, 16, 2.13)
    ratio = g.gear_ratio_from(v, cad, 2.13)
    assert ratio == pytest.approx(56 / 16)
    assert g.cadence_for(v, 56, 16, 2.13) == pytest.approx(v * 60 / (2.13 * 3.5))


def test_parse_cassette():
    assert g.parse_cassette("30 11,12-13") == [11, 12, 13, 30]
    assert g.CASSETTES["Ultegra 11-32 (12sp)"][-1] == 32
    with pytest.raises(ValueError):
        g.parse_cassette("11")


def test_pick_sprocket():
    cad = g.cadence_for(10.0, 56, 16, 2.13)
    idx, res = g.pick_sprocket_for_cadence(10.0, 56, CAS, cad, 2.13)
    assert CAS[idx] == 16 and res == pytest.approx(cad)
    idx, res = g.pick_sprocket_for_cadence(np.array([8.0, 14.0]), 56, CAS, 90.0, 2.13)
    assert len(idx) == 2 and np.all(np.abs(np.log(res / 90)) < 0.1)


def _synthetic(ring_true=56, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    sprockets = rng.choice(CAS[3:9], n)  # rider lives in the middle of the cassette
    cad = rng.normal(88, 4, n)
    speed = g.cadence_for(1.0, ring_true, sprockets, 2.13) * 0 + cad * (ring_true / sprockets) * 2.13 / 60
    return speed, cad


def test_recovers_best_ring():
    speed, cad = _synthetic()
    res = g.evaluate_chainrings(speed, cad, np.ones_like(speed), range(44, 68, 2), CAS, 2.13)
    best = res.sort_values("score", ascending=False).iloc[0]
    assert abs(best.chainring - 56) <= 2
    assert res.loc[res.chainring == 44, "score"].iloc[0] < best.score
    assert res.loc[res.chainring == 66, "score"].iloc[0] < best.score


def test_cadence_model_and_usage():
    speed, cad = _synthetic()
    w = np.ones_like(speed)
    mat = g.usage_matrix(speed, cad, w, [52, 56], CAS, 2.13)
    assert mat.shape == (2, len(CAS)) and (mat.sum(axis=1) <= 100.001).all()
    res = g.evaluate_from_cadence_model(speed, np.full_like(speed, 88.0), w, [48, 56, 64], CAS, 2.13)
    assert set(res.columns) >= {"chainring", "mid2_pct", "mid4_pct", "ends_pct", "out_of_range_pct",
                                "cadence_err_mean", "cadence_err_p90", "score"}


def test_out_of_range_detected():
    speed = np.full(100, 25.0)  # 90 km/h
    res = g.evaluate_chainrings(speed, np.full(100, 90.0), None, [40, 66], CAS, 2.13)
    assert (res.out_of_range_pct > 99).all()


FILES = [f for f in glob.glob(str(Path(__file__).parent.parent / "Fit_files" / "*")) if f.lower().endswith(".fit")]


@pytest.mark.skipif(not FILES, reason="no FIT files")
def test_real_files(capsys):
    from cycling_tools.fit_io import load_fit
    rides = [load_fit(f) for f in FILES]
    df = g.prepare_gearing_samples(rides, min_power=50)
    assert len(df) > 1000
    res = g.evaluate_chainrings(df.speed, df.cadence, df.weight_s, range(40, 68, 2), CAS)
    print(res.round(1).to_string())
    cm = g.fit_cadence_model(df)
    print(cm)
    best = res.sort_values("score", ascending=False).iloc[0].chainring
    assert 48 <= best <= 64
    assert 60 < cm["cadence_flat"] < 110
    climb = g.prepare_gearing_samples(rides, min_power=50, terrain="climb")
    assert len(climb) < len(df)
    both = g.evaluate_setups(df.speed, df.cadence, df.weight_s, g.default_setups("both"), CAS)
    print(both.sort_values("score", ascending=False).round(1).to_string())
    assert both.score.notna().all()


# ------------------------------------------------------------------ 1x / 2x setups
def test_setup_parsing():
    assert g.parse_setup("56/42").chainrings == (56, 42)
    assert g.parse_setup("58").chainrings == (58,) and not g.parse_setup("58").is_2x
    with pytest.raises(ValueError):
        g.parse_setup("50/40/30")


def test_1x_matches_old_behaviour():
    speed, cad = _synthetic(seed=3)
    speed = speed * np.random.default_rng(1).uniform(0.7, 1.4, len(speed))  # spread so ends/oor are exercised
    old = g.evaluate_chainrings(speed, cad, None, [44, 52, 56, 64], CAS, 2.13)
    new = g.evaluate_setups(speed, cad, None, [g.Setup((r,)) for r in [44, 52, 56, 64]], CAS, 2.13)
    for col in ["mid2_pct", "mid4_pct", "ends_pct", "out_of_range_pct", "cadence_err_mean", "cadence_err_p90", "score"]:
        np.testing.assert_allclose(new[col].to_numpy(), old[col].to_numpy(), atol=1e-6, err_msg=col)
    assert (new.cross_chain_pct == 0).all() and (new.aero_penalty_w == 0).all()
    i, s, c = g.pick_gear_for_setup(10.0, "56", CAS, 90.0, 2.13)
    i2, c2 = g.pick_sprocket_for_cadence(10.0, 56, CAS, 90.0, 2.13)
    assert (i, s) == (0, i2) and c == pytest.approx(c2)


def _mixed_ride(n=4000, seed=0):
    """Rider on 56/42: fast sections on 56 in the mid cassette, slow climbs on 42 in the mid cassette."""
    rng = np.random.default_rng(seed)
    spr = rng.choice(CAS[3:9], n)
    cad = rng.normal(88, 3, n)
    ring = np.where(np.arange(n) % 800 < 400, 56, 42)  # alternating blocks
    speed = cad * (ring / spr) * 2.13 / 60
    return speed, cad, ring


def test_2x_recovers_56_42_usage():
    speed, cad, ring = _mixed_ride()
    st = g.parse_setup("56/42")
    mat = g.usage_matrix_setup(speed, cad, None, st, CAS, 2.13)
    share56 = mat.loc[56].sum() / mat.to_numpy().sum()
    assert 0.35 < share56 < 0.75  # ~half the ride on each ring (overlap lets some samples go either way)
    assert mat.to_numpy().sum() > 99
    res = g.evaluate_setups(speed, cad, None, [g.Setup((56,)), st, g.Setup((42,))], CAS, 2.13)
    r = res.set_index("setup")
    assert r.loc["56/42", "out_of_range_pct"] < 1
    assert r.loc["56/42", "mid4_pct"] > r.loc["56", "mid4_pct"] + 10  # 1x56 pushes the climbs to the cassette end
    assert r.loc["42", "out_of_range_pct"] > 5  # 1x42 spins out on the fast half
    assert r.loc["56/42", "front_shifts_per_hour"] > 0
    assert r.loc["56/42", "aero_penalty_w"] > 0
    assert r.loc["56/42", "gear_range_pct"] > r.loc["56", "gear_range_pct"]
    free = g.evaluate_setups(speed, cad, None, [st], CAS, 2.13, weights={"aero_delta_cda": 0})
    assert free.aero_penalty_w.iloc[0] == 0


def test_cross_chain_flagged_and_avoided():
    st = g.parse_setup("56/42")
    m = g.cross_chain_mask(st, len(CAS), 3)
    assert m[0, -3:].all() and not m[0, :-3].any()  # big ring (idx 0): largest 3 crossed
    assert m[1, :3].all() and not m[1, 3:].any()   # small ring: smallest 3 crossed
    # only 56x30 gives the target cadence; the alternative (42x21) is ~7% off -> crossed gear kept and flagged
    v = float(g.cadence_for(1.0, 56, 30, 2.13) * 90 / g.cadence_for(1.0, 56, 30, 2.13))
    speed = float(g.cadence_for(1.0, 56, 30, 2.13)) * 0 + 90 * (56 / 30) * 2.13 / 60
    ri, si, cad = g.pick_gear_for_setup(speed, st, CAS, 90.0, 2.13)
    assert (st.chainrings[ri], CAS[si]) == (56, 30) and m[ri, si]
    seq = g.gear_sequence([speed] * 50, [90.0] * 50, None, st, CAS, 2.13)
    assert seq.crossed.all()
    # 42x21 exactly: prefer it over a (crossed) 56x27 that is only ~3.7% off
    speed2 = 90 * (42 / 21) * 2.13 / 60
    ri, si, _ = g.pick_gear_for_setup(speed2, st, CAS, 90.0, 2.13)
    assert (st.chainrings[ri], CAS[si]) == (42, 21)
    # equivalent gears: prefer big ring (56x14 == 42x10.5 -> choose via 56x14)
    ri, si, _ = g.pick_gear_for_setup(90 * (56 / 14) * 2.13 / 60, st, CAS, 90.0, 2.13)
    assert st.chainrings[ri] == 56


def test_front_shift_debounce():
    ring = np.array([0] * 30 + [1] * 3 + [0] * 30 + [1] * 30)
    assert g.count_front_shifts(ring, np.ones(len(ring)), 10) == 1
    assert g.count_front_shifts(ring, np.ones(len(ring)), 1) == 3


def test_2x_from_cadence_model_columns():
    speed, cad, _ = _mixed_ride()
    res = g.evaluate_setups_from_cadence_model(speed, 88.0, None, g.default_setups("both"), CAS, 2.13)
    assert len(res) == len(g.default_setups("both")) and res.score.notna().all()


def test_streamlit_page_smoke():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "pages" / "2_Gearing.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Recommended" in x.value for x in at.success)
    at.radio[0].set_value("2x only").run()
    assert not at.exception, [e.value for e in at.exception]
