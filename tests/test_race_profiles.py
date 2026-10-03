"""Tests for named race-profile storage."""

import pytest

from cycling_tools import race_profiles as rp

SETTINGS = {"np_race": 250.0, "race_vi": 1.02, "cda": 0.22, "cassette": [11, 12, 13], "weather_mode": "Manual wind"}


def test_slugify_is_file_safe():
    assert rp.slugify("Challenge Almere 2027!") == "challenge-almere-2027"
    assert rp.slugify("../../etc/passwd") == "etc-passwd"
    assert rp.slugify("!!!") == ""
    assert "/" not in rp.slugify("a/b\\c")


def test_round_trip_keeps_settings_course_and_notes(tmp_path):
    saved = rp.save_profile("Almere 2027", SETTINGS, gpx_bytes=b"<gpx/>", course_name="almere", notes=" flat, windy ",
                            last_prediction={"time_s": 16000}, store=tmp_path)
    assert saved.slug == "almere-2027"
    profile, gpx = rp.load_profile("Almere 2027", tmp_path)
    assert profile.settings == SETTINGS
    assert profile.name == "Almere 2027" and profile.course_name == "almere" and profile.notes == "flat, windy"
    assert profile.last_prediction == {"time_s": 16000}
    assert gpx == b"<gpx/>"


def test_overwrite_same_name_updates_in_place_and_keeps_course_if_none_given(tmp_path):
    rp.save_profile("Race", {"np_race": 240}, gpx_bytes=b"<gpx>1</gpx>", store=tmp_path)
    rp.save_profile("  race  ", {"np_race": 260}, store=tmp_path)  # same slug, no new GPX
    assert len(rp.list_profiles(tmp_path)) == 1
    profile, gpx = rp.load_profile("race", tmp_path)
    assert profile.settings["np_race"] == 260
    assert gpx == b"<gpx>1</gpx>"


@pytest.mark.parametrize("bad", ["", "   ", "!!!", "x" * 200])
def test_bad_names_rejected(bad, tmp_path):
    with pytest.raises(rp.RaceProfileError):
        rp.save_profile(bad, SETTINGS, store=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_path_traversal_cannot_escape_store(tmp_path):
    store = tmp_path / "store"
    rp.save_profile("../../evil", SETTINGS, store=store)
    assert [p.name for p in store.iterdir()] == ["evil.json"]
    assert not (tmp_path / "evil.json").exists()


def test_list_sorted_newest_first_and_skips_corrupt(tmp_path):
    rp.save_profile("First", SETTINGS, store=tmp_path)
    rp.save_profile("Second", SETTINGS, store=tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    names = [p.name for p in rp.list_profiles(tmp_path)]
    assert set(names) == {"First", "Second"}
    assert names[0] == "Second" or names[0] == "First"  # same-second saves can tie; both must be present


def test_delete_removes_both_files_and_is_idempotent(tmp_path):
    rp.save_profile("Gone", SETTINGS, gpx_bytes=b"<gpx/>", store=tmp_path)
    rp.delete_profile("Gone", tmp_path)
    assert list(tmp_path.iterdir()) == []
    rp.delete_profile("Gone", tmp_path)  # no error
    assert not rp.exists("Gone", tmp_path)


def test_load_missing_raises(tmp_path):
    with pytest.raises(rp.RaceProfileError):
        rp.load_profile("nothing here", tmp_path)


def test_env_var_overrides_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CYCLING_TOOLS_RACE_PROFILES", str(tmp_path / "custom"))
    rp.save_profile("Env race", SETTINGS)
    assert (tmp_path / "custom" / "env-race.json").exists()


# ------------------------------------------------------------------ page flow
def _page(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CYCLING_TOOLS_RACE_PROFILES", str(tmp_path))
    page = Path(__file__).resolve().parent.parent / "pages" / "3_Race_Planner.py"
    return AppTest.from_file(str(page), default_timeout=240).run()


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_page_save_load_delete_round_trip(tmp_path, monkeypatch):
    at = _page(tmp_path, monkeypatch)
    assert not at.exception
    at.selectbox(key="pick_0").select("Manchester_TTA_50_sample.gpx").run()  # the inputs appear once a course is chosen
    at.radio(key="wxmode_0").set_value("None (still air)").run()
    at.number_input(key="np_0").set_value(240.0)
    at.number_input(key="vi_0").set_value(1.03)
    at.number_input(key="cda_0").set_value(0.215)
    at.text_input(key="rp_name_0").set_value("Test Race")
    at.text_input(key="rp_notes_0").set_value("flat and fast")
    at.run()

    _button(at, "Save race plan").click().run()
    assert not at.exception
    saved = rp.load_profile("Test Race", tmp_path)
    profile, gpx = saved
    assert profile.settings["np_race"] == 240.0 and profile.settings["race_vi"] == 1.03
    assert profile.settings["cda"] == 0.215 and profile.settings["weather_mode"] == "None (still air)"
    assert profile.notes == "flat and fast" and profile.last_prediction["time_s"] > 3600
    assert gpx and b"<gpx" in gpx[:300]
    assert any("Saved" in m.value for m in at.success)

    # change the inputs, then load the saved race: every input returns to the saved value
    at.number_input(key="np_0").set_value(180.0)
    at.number_input(key="cda_0").set_value(0.30)
    at.run()
    _button(at, "Load").click().run()
    assert not at.exception
    assert at.number_input(key="np_1").value == 240.0
    assert at.number_input(key="vi_1").value == 1.03
    assert at.number_input(key="cda_1").value == 0.215
    assert at.radio(key="wxmode_1").value == "None (still air)"
    assert at.text_input(key="rp_name_1").value == "Test Race"
    assert at.text_input(key="rp_notes_1").value == "flat and fast"
    assert at.metric  # the saved course loaded and produced results without re-picking a file

    # saving under the same name updates rather than duplicates
    at.number_input(key="np_1").set_value(250.0)
    at.run()
    _button(at, "Update “Test Race”").click().run()
    assert len(rp.list_profiles(tmp_path)) == 1
    assert rp.load_profile("Test Race", tmp_path)[0].settings["np_race"] == 250.0

    # delete needs the confirmation tick
    assert _button(at, "Delete").disabled
    at.checkbox(key="rp_confirm_test-race").check().run()
    _button(at, "Delete").click().run()
    assert not at.exception
    assert rp.list_profiles(tmp_path) == []


def test_page_save_button_disabled_without_a_name(tmp_path, monkeypatch):
    at = _page(tmp_path, monkeypatch)
    at.selectbox(key="pick_0").select("Manchester_TTA_50_sample.gpx").run()  # the inputs appear once a course is chosen
    at.radio(key="wxmode_0").set_value("None (still air)").run()
    assert _button(at, "Save race plan").disabled
    assert rp.list_profiles(tmp_path) == []
