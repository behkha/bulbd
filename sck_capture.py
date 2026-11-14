"""
sck_capture.py - screen frames (macOS 12.3+) and system audio (13+) from
ScreenCaptureKit streams.

mss grabs the screen with CGWindowListCreateImage: every grab makes
WindowServer composite the display and makes tccd / trustd / syspolicyd
check the Screen Recording grant again, whatever the size of the box. A
stream is set up once: the permission is checked when it starts, the
system scales each frame down before handing it over, and a picture that
has not changed is reported as an `idle` sample with no pixels at all.

StreamGrabber keeps ScreenGrabber's contract - `frame()` is float32 RGB in
0..1, about `width` px wide, `box` is the captured area, `close()` - and
adds `pushes` and `pace()`, so a loop waits for the next frame instead of
grabbing on a timer. music_bulb.Capture picks between the two and falls
back to mss when this one cannot start or dies.

Every pyobjc call is in Kit. StreamGrabber takes any object with Kit's
methods, so the tests drive it with a fake on any platform.

What was checked on this Mac (macOS 27, pyobjc 13):
- No run loop is needed: the content query, start/stop handlers and the
  samples all arrive on ScreenCaptureKit's own dispatch threads, and None
  as the sample queue means its default queue.
- A block that returns anything but None aborts the whole process, and so
  does an exception raised in one; every callback here returns None and
  swallows its errors.
- An Objective-C subclass can be defined once per process; the output
  class is made on first use and shared.
- Rows are padded (190 px -> 768 bytes per row, 100 px -> 512).
- Leaving `colorSpaceName` unset delivers the display's own colour space
  ("Color LCD" on the built-in panel) - what mss returns - where sRGB or
  P3 would shift every value. The first frame's colour space is checked.
- `sourceRect` is in the display's own points, origin top left.

System audio rides on the same kit (macOS 13+): SystemAudio stands where a
soundcard microphone does and AudioStream where its recorder does, so the
audio loops read what the Mac is playing with no virtual cable and no
Multi-Output device. Checked on this Mac:
- A stream with only an audio output added delivers no pictures at all.
- The audio is 48000 Hz float32, planar (one block: all of the left
  channel, then all of the right), 960 frames every 20 ms, whatever
  `sampleRate` asks for - 44100 was asked and ignored - so Resampler brings
  it to the rate the loop wants.
- Silence is delivered as buffers of zeros at the same pace, not withheld.
- A buffer arrives 20-47 ms after its first sample was played.
"""

import atexit
import math
import sys
import threading
import time
import weakref

import numpy as np

COMPLETE = 0                       # SCFrameStatus: a new picture
IDLE = 1                           # nothing changed; the sample has no pixels
SYSTEM_AUDIO = "System audio"      # the input's name, on the desk and the CLI


class StreamDied(RuntimeError):
    """The stream stopped on its own: display gone, permission revoked..."""


def bgra_to_frame(buf, width, height, bytes_per_row):
    """A BGRA pixel buffer whose rows are `bytes_per_row` apart (padding
    included) -> float32 RGB in 0..1, shape (height, width, 3). The result
    is a copy, so the buffer may be unlocked right after."""
    if width <= 0 or height <= 0 or bytes_per_row < width * 4:
        raise ValueError("bad pixel buffer geometry")
    a = np.frombuffer(buf, dtype=np.uint8)
    if a.size < bytes_per_row * (height - 1) + width * 4:
        raise ValueError("pixel buffer shorter than its geometry")
    a = np.lib.stride_tricks.as_strided(
        a, (height, width, 4), (bytes_per_row, 4, 1), writeable=False)
    return a[..., 2::-1].astype(np.float32) / 255.0


def frame_size(box_w, box_h, width):
    """The size mss would produce for this box: ScreenGrabber stride-samples
    every n-th pixel, and the stream is asked for the same shape so the two
    are interchangeable (CapturePacer and MotionMeter count pixels)."""
    stride = max(1, int(box_w // max(width, 16)))
    return int(math.ceil(box_w / stride)), int(math.ceil(box_h / stride))


# ---------------------------------------------------------------------------
# the pyobjc side
# ---------------------------------------------------------------------------
_OUTPUT = None


def _output_class():
    """The SCStreamOutput / SCStreamDelegate class, defined once."""
    global _OUTPUT
    if _OUTPUT is None:
        import objc
        from Foundation import NSObject

        class BulbdStreamOutput(NSObject, protocols=[
                objc.protocolNamed("SCStreamOutput"),
                objc.protocolNamed("SCStreamDelegate")]):
            # an exception escaping either method aborts the process
            def stream_didOutputSampleBuffer_ofType_(self, stream, sample, kind):
                try:
                    self.on_sample(sample, kind)
                except Exception:
                    pass

            def stream_didStopWithError_(self, stream, error):
                try:
                    self.on_stop(error)
                except Exception:
                    pass

        _OUTPUT = BulbdStreamOutput
    return _OUTPUT


def _call(start, timeout):
    """Run an async ScreenCaptureKit call and wait for its handler.
    `start(handler)` makes the call; -> the handler's arguments."""
    done, box = threading.Event(), []

    def handler(*args):               # must return None: see the module notes
        box.append(args)
        done.set()

    start(handler)
    if not done.wait(timeout):
        raise TimeoutError("ScreenCaptureKit did not answer")
    return box[0]


class Kit:
    """The few ScreenCaptureKit calls a stream needs."""

    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("ScreenCaptureKit is macOS only")
        import CoreMedia
        import Quartz
        import ScreenCaptureKit
        self.S, self.CM, self.Q = ScreenCaptureKit, CoreMedia, Quartz
        self.Output = _output_class()

    def displays(self, timeout=3.0):
        """-> [(display, (x, y, w, h))] in global points, top-left origin -
        the coordinates mss uses for its monitors."""
        content, error = _call(
            lambda h: self.S.SCShareableContent
            .getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                False, True, h), timeout)
        if content is None:
            raise RuntimeError(f"no shareable content: {error}")
        out = []
        for d in content.displays():
            f = d.frame()
            out.append((d, (f.origin.x, f.origin.y, f.size.width, f.size.height)))
        return out

    def start(self, display, rect, size, fps, on_sample, on_stop, timeout=3.0):
        """Start a stream of `display` (or the `rect` of it, display points)
        at `size` pixels, no faster than `fps`. -> a handle for stop()."""
        S, Q = self.S, self.Q
        cfg = S.SCStreamConfiguration.alloc().init()
        cfg.setWidth_(int(size[0]))
        cfg.setHeight_(int(size[1]))
        cfg.setPixelFormat_(Q.kCVPixelFormatType_32BGRA)
        # colorSpaceName stays unset: the display's own space, as mss gives
        cfg.setMinimumFrameInterval_(self._interval(fps))
        cfg.setQueueDepth_(3)
        cfg.setShowsCursor_(False)             # CGWindowListCreateImage has none
        if cfg.respondsToSelector_(b"setPreservesAspectRatio:"):
            cfg.setPreservesAspectRatio_(False)    # no bars from a rounded size
        if rect is not None:
            cfg.setSourceRect_(Q.CGRectMake(*rect))
        flt = S.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
        out = self.Output.alloc().init()
        out.on_sample, out.on_stop = on_sample, on_stop
        stream = S.SCStream.alloc().initWithFilter_configuration_delegate_(flt, cfg, out)
        ok, err = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            out, S.SCStreamOutputTypeScreen, None, None)
        if not ok:
            raise RuntimeError(f"could not add the stream output: {err}")
        (error,) = _call(stream.startCaptureWithCompletionHandler_, timeout)
        if error is not None:
            raise RuntimeError(f"stream did not start: {error}")
        return {"stream": stream, "out": out, "cfg": cfg}

    def _interval(self, fps):
        ms = int(round(1000.0 / max(float(fps), 0.1)))
        return self.CM.CMTimeMake(ms, 1000)

    def configure(self, handle, fps):
        cfg = handle["cfg"]
        cfg.setMinimumFrameInterval_(self._interval(fps))

        def done(error):
            return None

        handle["stream"].updateConfiguration_completionHandler_(cfg, done)

    def start_audio(self, on_audio, on_stop, timeout=3.0):
        """Start a stream that carries only what the Mac is playing.
        `on_audio(sample)` gets each buffer. -> a handle for stop()."""
        S = self.S
        shown = self.displays(timeout)
        if not shown:
            raise RuntimeError("no display to attach the audio stream to")
        main = self.Q.CGMainDisplayID()
        display = next((d for d, _ in shown if d.displayID() == main), shown[0][0])
        cfg = S.SCStreamConfiguration.alloc().init()
        cfg.setWidth_(2)                       # a picture is never asked for:
        cfg.setHeight_(2)                      # no screen output is added
        cfg.setMinimumFrameInterval_(self.CM.CMTimeMake(10, 1))
        cfg.setCapturesAudio_(True)
        cfg.setChannelCount_(2)
        cfg.setExcludesCurrentProcessAudio_(True)
        flt = S.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
        audio = S.SCStreamOutputTypeAudio

        def on_sample(sample, kind):
            if kind == audio:
                on_audio(sample)

        out = self.Output.alloc().init()
        out.on_sample, out.on_stop = on_sample, on_stop
        stream = S.SCStream.alloc().initWithFilter_configuration_delegate_(flt, cfg, out)
        ok, err = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            out, audio, None, None)
        if not ok:
            raise RuntimeError(f"could not add the audio output: {err}")
        (error,) = _call(stream.startCaptureWithCompletionHandler_, timeout)
        if error is not None:
            raise RuntimeError(f"stream did not start: {error}")
        return {"stream": stream, "out": out, "cfg": cfg}

    def read_audio(self, sample):
        """-> (float32 pcm of shape (frames, channels), its sample rate),
        or None for a buffer with nothing in it."""
        CM = self.CM
        frames = CM.CMSampleBufferGetNumSamples(sample)
        fmt = CM.CMSampleBufferGetFormatDescription(sample)
        block = CM.CMSampleBufferGetDataBuffer(sample)
        if not frames or fmt is None or block is None:
            return None
        # AudioStreamBasicDescription: rate, format, flags, bytes per packet,
        # frames per packet, bytes per frame, channels, bits per channel
        d = CM.CMAudioFormatDescriptionGetStreamBasicDescription(fmt)
        rate, flags, channels, bits = d[0], d[2], d[6], d[7]
        if not flags & 1 or bits != 32:        # kAudioFormatFlagIsFloat
            raise ValueError("system audio is not 32-bit float")
        status, data = CM.CMBlockBufferCopyDataBytes(
            block, 0, CM.CMBlockBufferGetDataLength(block), None)
        if status != 0:
            return None
        # kAudioFormatFlagIsNonInterleaved = 32
        return pcm_frames(data, frames, channels, not flags & 32), float(rate)

    def stop(self, handle, timeout=2.0):
        """Stop the stream and wait for it to say so. -> True if it did."""
        out = handle["out"]
        out.on_sample = out.on_stop = _ignore
        try:
            _call(handle["stream"].stopCaptureWithCompletionHandler_, timeout)
            return True
        except TimeoutError:
            return False

    def read(self, sample, convert):
        """-> (status, convert(buf, w, h, bytes_per_row) or None)."""
        CM, Q, S = self.CM, self.Q, self.S
        att = CM.CMSampleBufferGetSampleAttachmentsArray(sample, False)
        status = att[0].get(S.SCStreamFrameInfoStatus) if att else None
        if status != COMPLETE:
            return status, None
        pb = CM.CMSampleBufferGetImageBuffer(sample)
        if pb is None:
            return status, None
        Q.CVPixelBufferLockBaseAddress(pb, Q.kCVPixelBufferLock_ReadOnly)
        try:
            w, h = Q.CVPixelBufferGetWidth(pb), Q.CVPixelBufferGetHeight(pb)
            bpr = Q.CVPixelBufferGetBytesPerRow(pb)
            base = Q.CVPixelBufferGetBaseAddress(pb)
            return status, convert(base.as_buffer(bpr * h), w, h, bpr)
        finally:
            Q.CVPixelBufferUnlockBaseAddress(pb, Q.kCVPixelBufferLock_ReadOnly)

    def color_space(self, sample):
        """The first frame's colour space name, for the check in the notes."""
        pb = self.CM.CMSampleBufferGetImageBuffer(sample)
        if pb is None:
            return None
        cs = self.Q.CVBufferCopyAttachment(pb, self.Q.kCVImageBufferCGColorSpaceKey, None)
        cs = cs[0] if isinstance(cs, tuple) else cs
        return None if cs is None else str(cs)


def _ignore(*_):
    return None


# ---------------------------------------------------------------------------
# the grabber
# ---------------------------------------------------------------------------
_LIVE = weakref.WeakSet()


@atexit.register
def close_all():
    """No stream outlives the process: the recording indicator goes off."""
    for grab in list(_LIVE):
        grab.close()


def audio_supported():
    """Could a system audio stream exist here? macOS 13+ with pyobjc's
    ScreenCaptureKit installed. The Screen Recording grant is the caller's
    question (music_bulb.system_audio_blocked). Imports no pyobjc."""
    if sys.platform != "darwin":
        return False
    import importlib.util
    import platform
    try:
        if int(platform.mac_ver()[0].split(".")[0]) < 13:
            return False
        return all(importlib.util.find_spec(m) is not None
                   for m in ("ScreenCaptureKit", "CoreMedia", "Quartz"))
    except Exception:
        return False


def find_display(displays, box):
    """The display whose frame is `box` {left, top, width, height}, else the
    one holding its centre. Raises RuntimeError if none does."""
    for d, (x, y, w, h) in displays:
        if abs(x - box["left"]) < 1 and abs(y - box["top"]) < 1 \
                and abs(w - box["width"]) < 1 and abs(h - box["height"]) < 1:
            return d, (x, y, w, h)
    cx, cy = box["left"] + box["width"] / 2, box["top"] + box["height"] / 2
    for d, (x, y, w, h) in displays:
        if x <= cx < x + w and y <= cy < y + h:
            return d, (x, y, w, h)
    raise RuntimeError("the display is not among ScreenCaptureKit's")


class StreamGrabber:
    """ScreenGrabber's contract on a ScreenCaptureKit stream.

    `display` is the mss monitor dict of the display to capture, `region`
    an absolute (x, y, w, h) inside it or None. Frames are copied out of the
    sample on ScreenCaptureKit's thread and kept as the latest one; `frame()`
    returns it without waiting. `pace()` is the wait between two frames: it
    ends on a new picture, on `wake` (a control moved) or at the deadline,
    whichever comes first.
    """

    pushes = True
    # A picture wakes the loop only when it moved as much as CapturePacer
    # counts as motion. The stream reports every change - a blinking cursor,
    # a spinner - and waking on each would hold a still picture at the full
    # rate; smaller changes are kept and read on the pacer's own clock.
    PIXEL, SHARE = 0.04, 0.002

    def __init__(self, display, region=None, width=192, fps=10, kit=None,
                 first_frame=3.0):
        self.kit = kit if kit is not None else Kit()
        if region:
            x, y, w, h = region
            self.box = {"left": x, "top": y, "width": w, "height": h}
        else:
            self.box = dict(display)
        disp, (dx, dy, _dw, _dh) = find_display(self.kit.displays(), display)
        rect = None
        if region:
            rect = (self.box["left"] - dx, self.box["top"] - dy,
                    self.box["width"], self.box["height"])
        self.size = frame_size(self.box["width"], self.box["height"], width)
        self.fps = fps
        self.seq = self._seen = 0
        self.delivered = self.idle = 0       # complete / idle samples, for the bench
        self.color_space = None
        self.error = None
        self.wake = None
        self._frame = None
        self._ref = None                     # the picture the last wake was for
        self._lock = threading.Lock()
        self._first = threading.Event()
        self._handle = self.kit.start(disp, rect, self.size, fps,
                                      self._on_sample, self._on_stop)
        _LIVE.add(self)
        if not self._first.wait(first_frame) or self._frame is None:
            err = self.error
            self.close()
            raise err or RuntimeError("the stream sent no picture")

    # -- ScreenCaptureKit's thread ------------------------------------------
    def _on_sample(self, sample, kind):
        status, frame = self.kit.read(sample, bgra_to_frame)
        if frame is None:
            if status == IDLE:
                self.idle += 1
            return
        if self.color_space is None:
            self.color_space = self.kit.color_space(sample)
        ref = self._ref
        moved = (ref is None or ref.shape != frame.shape
                 or float((np.abs(frame - ref).max(axis=2) > self.PIXEL).mean()) > self.SHARE)
        with self._lock:
            self._frame = frame
            self.delivered += 1
            if moved:
                self._ref = frame
                self.seq += 1
        self._first.set()
        wake = self.wake
        if moved and wake is not None:
            wake.set()

    def _on_stop(self, error):
        self.error = StreamDied(f"the screen stream stopped: {error}")
        self._first.set()
        wake = self.wake
        if wake is not None:
            wake.set()

    # -- the loop's thread --------------------------------------------------
    def frame(self):
        """The latest picture, however small its change."""
        if self.error is not None:
            raise self.error
        with self._lock:
            self._seen = self.seq
            return self._frame

    def pace(self, started, base, period, wake=None):
        """Wait from `started` until a new picture arrives (but not before
        `base` has passed) or until `period` has. -> True if `wake` ended
        the wait instead - a control moved."""
        if wake is None:
            wake = self.wake if self.wake is not None else threading.Event()
        self.wake = wake
        wake.clear()
        seen = self._seen
        slack = started + base - time.monotonic()
        if slack > 0:
            time.sleep(slack)
        while True:
            if self.seq != seen or self.error is not None:
                return False
            left = started + period - time.monotonic()
            if left <= 0:
                return False
            if wake.wait(left):
                wake.clear()
                return self.seq == seen and self.error is None

    def set_fps(self, fps):
        if fps != self.fps and self._handle is not None:
            self.fps = fps
            try:
                self.kit.configure(self._handle, fps)
            except Exception:
                pass

    def close(self):
        """Stop the stream. Safe to call twice and from any thread."""
        with self._lock:
            handle, self._handle = self._handle, None
        _LIVE.discard(self)
        if handle is not None:
            try:
                self.kit.stop(handle)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# system audio
# ---------------------------------------------------------------------------
def pcm_frames(data, frames, channels, interleaved):
    """Raw float32 samples -> shape (frames, channels). A planar buffer holds
    each channel whole, one after the other."""
    a = np.frombuffer(data, dtype=np.float32)
    if frames <= 0 or channels <= 0 or a.size < frames * channels:
        raise ValueError("audio buffer shorter than its format")
    a = a[:frames * channels]
    return a.reshape(frames, channels) if interleaved else a.reshape(channels, frames).T


def fit_channels(pcm, channels):
    """`pcm` (frames, n) with the channel count the loop asked for."""
    have = pcm.shape[1]
    if have == channels:
        return pcm
    if channels == 1:
        return pcm.mean(axis=1, keepdims=True)
    if have == 1:
        return np.repeat(pcm, channels, axis=1)
    return pcm[:, :channels] if have > channels else np.pad(pcm, ((0, 0), (0, channels - have)))


class Resampler:
    """A stream of (frames, channels) blocks from `src` Hz to `dst` Hz by
    linear interpolation, continuous across blocks. Enough for a ten-band
    analyser: the loss is a fixed tilt (3 dB at 16 kHz going 48000 -> 44100),
    and every band is judged against its own recent peak.

    It runs fifty times a second for as long as a session does, so the work
    per block is kept to the interpolation itself: the position is counted
    in whole 1/dst frames (exact, and 960 frames at 48000 -> 44100 end where
    they began), which lets the indices and weights of a block be worked out
    once and reused; and digital silence - what the stream sends while
    nothing plays - is answered with zeros without interpolating.
    """

    PLANS = 64                        # block shapes remembered

    def __init__(self, src, dst):
        self.src, self.dst = float(src), float(dst)
        self._s, self._d = int(round(self.src)), int(round(self.dst))
        self.phase = 0                # the next output, in 1/dst input frames
        self.tail = None              # the last input frame of the block before
        self._quiet = False           # that block was all zeros
        self._plans = {}

    def _plan(self, last):
        """Where the outputs of a block whose last frame is `last` fall:
        (count, left index, right index, right weight, left weight, the
        phase after it)."""
        key = (last, self.phase)
        plan = self._plans.get(key)
        if plan is None:
            s, d = self._s, self._d
            count = max(0, -((self.phase - last * d) // s))
            at = self.phase + s * np.arange(count, dtype=np.int64)
            i = at // d
            frac = ((at - i * d) / d).astype(np.float32)[:, None]
            plan = (count, i, i + 1, frac, 1 - frac, self.phase + count * s - last * d)
            if len(self._plans) < self.PLANS:
                self._plans[key] = plan
        return plan

    def feed(self, pcm):
        if self._s == self._d:
            return pcm
        was_quiet, self._quiet = self._quiet, not pcm.any()
        tail, self.tail = self.tail, pcm[-1:]
        count, i, j, frac, keep, self.phase = self._plan(
            len(pcm) - (1 if tail is None else 0))
        if self._quiet and (was_quiet or tail is None):
            return np.zeros((count, pcm.shape[1]), dtype=np.float32)
        if tail is not None:
            pcm = np.concatenate([tail, pcm])
        return pcm[i] * keep + pcm[j] * frac


class AudioStream:
    """What a soundcard recorder gives a loop - `record(numframes)` blocks
    until that many frames are there and returns float32 of shape
    (numframes, channels) - fed by the system audio stream.

    Buffers are converted on ScreenCaptureKit's thread and queued; at most
    BACKLOG seconds are kept, so a loop that stopped reading (a reconnect)
    comes back to the present instead of working through the past. A stream
    that stalls costs a block of silence per GRACE rather than a hung loop;
    one that stops is started again, at most once per RESTART seconds.
    """

    BACKLOG = 0.5
    GRACE = 0.25
    RESTART = 5.0

    def __init__(self, samplerate, channels=2, kit=None):
        self.rate, self.channels = float(samplerate), int(channels)
        self.kit = kit
        self.error = None
        self.dropped = 0                       # frames the backlog cap threw away
        self._cap = int(self.BACKLOG * self.rate)
        self._chunks, self._have = [], 0
        self._want = 0                         # frames the waiting record() needs
        self._cond = threading.Condition()
        self._resample = None
        self._handle = None
        self._started = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, *exc):
        self.close()

    def _start(self):
        try:
            if self.kit is None:
                self.kit = Kit()
            self.error = None
            self._started = time.monotonic()
            self._handle = self.kit.start_audio(self._on_audio, self._on_stop)
        except Exception as e:
            raise RuntimeError(f"{SYSTEM_AUDIO} did not start: {e}") from e
        _LIVE.add(self)

    # -- ScreenCaptureKit's thread ------------------------------------------
    def _on_audio(self, sample):
        got = self.kit.read_audio(sample)
        if got is None:
            return
        pcm, rate = got
        if self._resample is None or self._resample.src != rate:
            self._resample = Resampler(rate, self.rate)
        pcm = fit_channels(self._resample.feed(pcm), self.channels)
        if not len(pcm):
            return
        with self._cond:
            self._chunks.append(pcm)
            self._have += len(pcm)
            while self._have - len(self._chunks[0]) >= self._cap:
                old = self._chunks.pop(0)
                self._have -= len(old)
                self.dropped += len(old)
            if self._have >= self._want:       # else the loop would wake for nothing
                self._cond.notify()

    def _on_stop(self, error):
        with self._cond:
            self.error = StreamDied(f"the audio stream stopped: {error}")
            self._cond.notify()

    # -- the loop's thread --------------------------------------------------
    def record(self, numframes):
        numframes = int(numframes)
        self._cap = max(self._cap, 2 * numframes)
        deadline = time.monotonic() + numframes / self.rate + self.GRACE
        with self._cond:
            self._want = numframes
            while self._have < numframes and self.error is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(left)
            died = self.error if self._have < numframes else None
            out = self._take(numframes)
        if died is not None:
            self._restart(died)
        return out

    def _take(self, n):
        parts, got = [], 0
        while got < n and self._chunks:
            chunk = self._chunks[0]
            if len(chunk) <= n - got:
                parts.append(self._chunks.pop(0))
            else:
                parts.append(chunk[:n - got])
                self._chunks[0] = chunk[n - got:]
            got += len(parts[-1])
        self._have -= got
        if got < n:                            # a stall: the rest is silence
            parts.append(np.zeros((n - got, self.channels), dtype=np.float32))
        return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)

    def _restart(self, died):
        """The stream stopped (the display went, the Mac slept): one new one,
        unless the last start is too recent to be anything but a loop."""
        if self._handle is None:
            raise died                          # closed, or never started
        if time.monotonic() - self._started < self.RESTART:
            raise died
        self._stop()
        try:
            self._start()
        except RuntimeError as e:
            raise StreamDied(f"{died}; {e}") from e

    def _stop(self):
        with self._cond:
            handle, self._handle = self._handle, None
        _LIVE.discard(self)
        if handle is not None:
            try:
                self.kit.stop(handle)
            except Exception:
                pass

    def close(self):
        """Stop the stream. Safe to call twice and from any thread."""
        self._stop()
        with self._cond:
            self._cond.notify_all()


class SystemAudio:
    """What the Mac is playing, where the loops expect a soundcard
    microphone: a `name` and `recorder()`. Nothing starts until the
    recorder is entered."""

    name = SYSTEM_AUDIO

    def __init__(self, kit=None):
        self.kit = kit

    def recorder(self, samplerate, channels=2, **_):
        return AudioStream(samplerate, channels, kit=self.kit)
