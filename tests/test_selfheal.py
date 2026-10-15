import pytest

from ytscribe import ytdlp_runner as yt


@pytest.fixture(autouse=True)
def reset():
    yt._state.update(upgrade_attempted=False, auto_update=True)


class Boom(Exception):
    pass


def make_fn(failures, msg):
    calls = {"n": 0}

    def fn(ydl):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise Boom(msg)
        return "ok"

    return fn, calls


def patch_call(monkeypatch):
    monkeypatch.setattr(yt, "_call", lambda fn, opts, ctx: fn(None))


def test_extractor_failure_upgrades_and_retries_once(monkeypatch):
    patch_call(monkeypatch)
    upgrades = []
    monkeypatch.setattr(yt, "_upgrade", lambda: upgrades.append(1) or True)
    fn, calls = make_fn(1, "ERROR: Unable to extract nsig function")
    assert yt.run(fn) == "ok"
    assert len(upgrades) == 1 and calls["n"] == 2


def test_upgrade_happens_at_most_once_per_run(monkeypatch):
    patch_call(monkeypatch)
    upgrades = []
    monkeypatch.setattr(yt, "_upgrade", lambda: upgrades.append(1) or True)
    fn, _ = make_fn(99, "Unable to extract")
    with pytest.raises(yt.YtdlpFailure):
        yt.run(fn)
    fn2, calls2 = make_fn(99, "Unable to extract")
    with pytest.raises(yt.YtdlpFailure):
        yt.run(fn2)
    assert len(upgrades) == 1 and calls2["n"] == 1  # second video: no upgrade, no retry


@pytest.mark.parametrize("msg", ["Private video. Sign in", "Video unavailable", "Sign in to confirm your age"])
def test_permanent_errors_never_upgrade(monkeypatch, msg):
    patch_call(monkeypatch)
    monkeypatch.setattr(yt, "_upgrade", lambda: pytest.fail("must not upgrade"))
    fn, calls = make_fn(99, msg)
    with pytest.raises(yt.VideoUnavailable):
        yt.run(fn)
    assert calls["n"] == 1


def test_auto_update_disabled(monkeypatch):
    patch_call(monkeypatch)
    yt.configure(auto_update=False)
    monkeypatch.setattr(yt, "_upgrade", lambda: pytest.fail("must not upgrade"))
    fn, _ = make_fn(99, "Unable to extract")
    with pytest.raises(yt.YtdlpFailure):
        yt.run(fn)
