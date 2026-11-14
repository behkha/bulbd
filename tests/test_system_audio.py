"""System audio: the ScreenCaptureKit audio stream standing where a soundcard
microphone does (with a fake Kit, so this runs anywhere without pyobjc), the
resampling, and how the input is picked and listed."""
import threading
import time

import numpy as np
import pytest

import music_bulb
import music_bulb_dash as dash
import sck_capture
from music_bulb import SYSTEM_AUDIO, Analyser, pick_recorder
from sck_capture import (AudioStream, Resampler, StreamDied, SystemAudio,
                         fit_channels, pcm_frames)


# ------------------------------------------------------------------ buffers
def test_a_planar_buffer_is_each_channel_whole_one_after_the_other():
    left, right = np.arange(4, dtype=np.float32), -np.arange(4, dtype=np.float32)
    pcm = pcm_frames(np.concatenate([left, right]).tobytes(), 4, 2, interleaved=False)
    assert pcm.shape == (4, 2)
    assert np.array_equal(pcm[:, 0], left) and np.array_equal(pcm[:, 1], right)


def test_an_interleaved_buffer_alternates_the_channels():
    data = np.array([0, 10, 1, 11, 2, 12], dtype=np.float32).tobytes()
    pcm = pcm_frames(data, 3, 2, interleaved=True)
    assert np.array_equal(pcm, [[0, 10], [1, 11], [2, 12]])


@pytest.mark.parametrize("frames, channels, size", [(4, 2, 28), (0, 2, 32), (4, 0, 32)])
def test_a_buffer_that_does_not_fit_its_format_is_refused(frames, channels, size):
    with pytest.raises(ValueError):
        pcm_frames(bytes(size), frames, channels, False)


def test_the_loop_gets_the_channel_count_it_asked_for():
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    assert fit_channels(stereo, 2) is stereo
    assert np.array_equal(fit_channels(stereo, 1), [[2.0], [3.0]])
    assert np.array_equal(fit_channels(stereo[:, :1], 2), [[1.0, 1.0], [2.0, 2.0]])


# ---------------------------------------------------------------- resampling
def tone(freq, rate, frames, start=0):
    t = (start + np.arange(frames)) / rate
    x = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([x, x], axis=1)


def test_resampling_keeps_the_pitch_and_the_length():
    r = Resampler(48000, 44100)
    out = np.concatenate([r.feed(tone(1000, 48000, 960, i * 960)) for i in range(100)])
    assert abs(len(out) - 88200) <= 1                    # two seconds, to the frame
    spec = np.abs(np.fft.rfft(out[:44100, 0]))
    assert int(spec.argmax()) == 1000                    # 1 Hz bins


def test_resampling_is_continuous_across_blocks():
    whole = Resampler(48000, 44100).feed(tone(440, 48000, 9600))
    r = Resampler(48000, 44100)
    parts = np.concatenate([r.feed(tone(440, 48000, 960, i * 960)) for i in range(10)])
    n = min(len(whole), len(parts))
    assert abs(len(whole) - len(parts)) <= 1
    assert np.allclose(whole[:n], parts[:n], atol=1e-4)


def test_the_same_rate_passes_straight_through():
    x = tone(440, 44100, 512)
    assert Resampler(44100, 44100).feed(x) is x


# ------------------------------------------------------------------- stream
class FakeKit:
    """Kit's audio half: hands the stream whatever the test pushes."""

    def __init__(self, rate=44100.0, fail=False):
        self.rate, self.fail = rate, fail
        self.started = self.stopped = 0
        self.on_audio = self.on_stop = None

    def start_audio(self, on_audio, on_stop):
        if self.fail:
            raise RuntimeError("stream did not start: nope")
        self.started += 1
        self.on_audio, self.on_stop = on_audio, on_stop
        return {"n": self.started}

    def read_audio(self, sample):
        return None if sample is None else (sample, self.rate)

    def stop(self, handle):
        self.stopped += 1
        return True

    def push(self, pcm):
        self.on_audio(np.asarray(pcm, dtype=np.float32))


def ramp(frames, start=0):
    x = (start + np.arange(frames)).astype(np.float32)
    return np.stack([x, -x], axis=1)


def test_record_returns_exactly_the_frames_asked_for_in_order():
    kit = FakeKit()
    with SystemAudio(kit).recorder(samplerate=44100, channels=2) as rec:
        for i in range(3):
            kit.push(ramp(882, i * 882))                 # 20 ms buffers
        a, b = rec.record(numframes=1024), rec.record(numframes=1024)
    assert a.shape == b.shape == (1024, 2) and a.dtype == np.float32
    assert np.array_equal(np.concatenate([a, b])[:, 0], np.arange(2048))
    assert kit.started == kit.stopped == 1


def test_record_waits_for_the_buffer_that_completes_the_block():
    kit = FakeKit()
    with AudioStream(44100, 2, kit=kit) as rec:
        kit.push(ramp(882))
        threading.Timer(0.05, lambda: kit.push(ramp(882, 882))).start()
        t0 = time.monotonic()
        out = rec.record(numframes=1024)
        assert 0.03 < time.monotonic() - t0 < 0.2
    assert np.array_equal(out[:, 0], np.arange(1024))


def test_a_stalled_stream_gives_silence_instead_of_hanging_the_loop():
    kit = FakeKit()
    with AudioStream(44100, 2, kit=kit) as rec:
        rec.GRACE = 0.02
        kit.push(ramp(100, 1))
        out = rec.record(numframes=1024)
    assert out.shape == (1024, 2)
    assert np.array_equal(out[:100, 0], np.arange(1, 101)) and not out[100:].any()


def test_a_loop_that_stopped_reading_comes_back_to_the_present():
    kit = FakeKit()
    with AudioStream(44100, 2, kit=kit) as rec:
        for i in range(200):                             # four seconds unread
            kit.push(ramp(882, i * 882))
        assert rec._have <= rec._cap + 882 and rec.dropped > 0
        out = rec.record(numframes=1024)
    assert out[0, 0] >= 200 * 882 - rec._cap - 882       # the newest half second


def test_the_stream_is_resampled_to_the_rate_the_loop_asked_for():
    kit = FakeKit(rate=48000.0)
    with AudioStream(44100, 2, kit=kit) as rec:
        for i in range(10):
            kit.push(tone(1000, 48000, 960, i * 960))
        assert abs(rec._have - 8820) <= 1
        out = rec.record(numframes=4410)
    assert int(np.abs(np.fft.rfft(out[:, 0])).argmax()) == 100   # 10 Hz bins


def test_an_empty_buffer_is_skipped():
    kit = FakeKit()
    with AudioStream(44100, 2, kit=kit) as rec:
        kit.on_audio(None)
        assert rec._have == 0


def test_a_stream_that_stops_is_started_again():
    kit = FakeKit()
    with AudioStream(44100, 2, kit=kit) as rec:
        rec.RESTART = 0.0
        kit.push(ramp(500))
        kit.on_stop("display gone")
        out = rec.record(numframes=1024)                 # what was left, then silence
        assert np.array_equal(out[:500, 0], np.arange(500)) and not out[500:].any()
        assert kit.started == 2 and rec.error is None
        kit.push(ramp(1024, 7))
        assert rec.record(numframes=1024)[0, 0] == 7
    assert kit.stopped == 2


def test_a_stream_that_keeps_stopping_ends_the_session():
    kit = FakeKit()
    with AudioStream(44100, 2, kit=kit) as rec:
        kit.on_stop("permission revoked")                # right after its start
        with pytest.raises(StreamDied, match="permission revoked"):
            rec.record(numframes=1024)


def test_what_is_buffered_is_played_out_before_a_stop_counts():
    kit = FakeKit()
    with AudioStream(44100, 2, kit=kit) as rec:
        kit.push(ramp(2048))
        kit.on_stop("gone")
        assert rec.record(numframes=1024)[0, 0] == 0     # no error yet


def test_a_stream_that_cannot_start_says_so_by_name():
    with pytest.raises(RuntimeError, match="System audio did not start: .*nope"):
        with AudioStream(44100, 2, kit=FakeKit(fail=True)):
            pass


def test_no_audio_stream_outlives_the_process():
    kit = FakeKit()
    rec = AudioStream(44100, 2, kit=kit)
    rec.__enter__()
    sck_capture.close_all()
    assert kit.stopped == 1
    rec.close()                                          # twice is safe
    assert kit.stopped == 1


def test_the_analyser_reads_a_block_from_the_stream():
    kit = FakeKit(rate=48000.0)
    an = Analyser()
    with AudioStream(music_bulb.SAMPLERATE, 2, kit=kit) as rec:
        for i in range(4):
            kit.push(0.5 * tone(220, 48000, 960, i * 960))
        f = an.feed(rec.record(numframes=music_bulb.BLOCK))
    assert f["rms"] == pytest.approx(0.5 / np.sqrt(2), rel=0.05)


# ------------------------------------------------------------------ picking
@pytest.fixture
def no_soundcard(monkeypatch):
    """pick_recorder must not reach soundcard on the system audio paths."""
    import sys
    monkeypatch.setitem(sys.modules, "soundcard", None)   # import -> ImportError


def test_automatic_is_system_audio_where_it_can_be_captured(monkeypatch, no_soundcard):
    monkeypatch.setattr(music_bulb, "system_audio_blocked", lambda: None)
    rec = pick_recorder(None, quiet=True)
    assert isinstance(rec, SystemAudio) and rec.name == SYSTEM_AUDIO


def test_system_audio_can_be_named_whatever_the_case(monkeypatch, no_soundcard):
    monkeypatch.setattr(music_bulb, "system_audio_blocked", lambda: None)
    assert isinstance(pick_recorder("system AUDIO"), SystemAudio)


def test_naming_it_without_the_permission_says_what_is_missing(monkeypatch, no_soundcard):
    monkeypatch.setattr(sck_capture, "audio_supported", lambda: True)
    monkeypatch.setattr(music_bulb, "screen_permission", lambda: False)
    with pytest.raises(RuntimeError, match="Screen Recording"):
        pick_recorder(SYSTEM_AUDIO)
    with pytest.raises(RuntimeError, match="Screen Recording"):
        dash.open_recorder(SYSTEM_AUDIO)                 # not "is not connected"


def test_without_it_automatic_falls_through_to_the_devices(monkeypatch):
    import sys
    import types
    mic = types.SimpleNamespace(name="MacBook Pro Microphone")

    def none(*a, **k):
        raise RuntimeError("no such device")

    sc = types.SimpleNamespace(
        get_microphone=none, all_microphones=lambda **k: [mic],
        default_speaker=lambda: types.SimpleNamespace(name="Speakers", id=1),
        default_microphone=lambda: mic)
    monkeypatch.setitem(sys.modules, "soundcard", sc)
    monkeypatch.setattr(music_bulb, "system_audio_blocked", lambda: "no permission")
    assert pick_recorder(None, quiet=True) is mic


def test_it_is_blocked_off_macos_and_without_the_permission(monkeypatch):
    monkeypatch.setattr(sck_capture, "audio_supported", lambda: False)
    assert music_bulb.system_audio_blocked()
    monkeypatch.setattr(sck_capture, "audio_supported", lambda: True)
    monkeypatch.setattr(music_bulb, "screen_permission", lambda: True)
    assert music_bulb.system_audio_blocked() is None
    monkeypatch.setattr(music_bulb, "screen_permission", lambda: False)
    assert "Screen Recording" in music_bulb.system_audio_blocked()


# --------------------------------------------------------------------- desk
def test_the_desk_lists_it_first_and_as_the_automatic_choice(monkeypatch):
    import sys
    import types
    mics = [types.SimpleNamespace(name="BlackHole 2ch")]
    monkeypatch.setitem(sys.modules, "soundcard",
                        types.SimpleNamespace(all_microphones=lambda **k: mics))
    monkeypatch.setattr(dash, "system_audio_blocked", lambda: None)
    monkeypatch.setattr(music_bulb, "system_audio_blocked", lambda: None)
    assert dash.audio_inputs() == {"inputs": [SYSTEM_AUDIO, "BlackHole 2ch"],
                                   "auto": SYSTEM_AUDIO}


def test_the_desk_leaves_it_out_where_it_cannot_be_captured(monkeypatch):
    import sys
    import types
    mic = types.SimpleNamespace(name="BlackHole 2ch")
    monkeypatch.setitem(sys.modules, "soundcard", types.SimpleNamespace(
        all_microphones=lambda **k: [mic], default_microphone=lambda: mic,
        get_microphone=lambda *a, **k: mic,
        default_speaker=lambda: types.SimpleNamespace(name="x", id=1)))
    monkeypatch.setattr(dash, "system_audio_blocked", lambda: "no")
    monkeypatch.setattr(music_bulb, "system_audio_blocked", lambda: "no")
    assert dash.audio_inputs() == {"inputs": ["BlackHole 2ch"], "auto": "BlackHole 2ch"}
