<div align="center">

<img src="assets/sticker.gif" alt="a shiba in sunglasses, dancing" width="220">

# bulbd

**Make a Xiaomi bulb dance to your music, your films, or both at once.**

DSP in, hue out — plus a built-in Xiaomi cloud token extractor.

`music` · `screen` · `fusion` · local web desk · miio token extraction

<br>

<img src="assets/desk-runtime.png" alt="the bulbd desk in screen mode: live colour field, hue histogram, and the capture controls" width="820">

<sub>Screen mode, mid-run. The address in the top bar is this machine's — yours will differ.</sub>

</div>

---

## What this is

A Mi Smart LED Bulb Essential (White & Color, `yeelink.light.color5`) driven
over the Yeelight LAN API at 25 updates a second. Three ways to drive it:

| Mode | Source | Behaviour |
|---|---|---|
| **music** | microphone or loopback PCM | onset detection steps the colour through palette anchors |
| **screen** | whatever is on a display | dominant on-screen colour, interpolated, snapping on scene cuts |
| **fusion** | both | the display picks hue and saturation, the audio drives brightness |

Music mode never drifts the hue freely — colour moves in discrete palette
steps on onsets. Screen mode deliberately does the opposite, because an
ambient lamp tracking a film should follow the picture rather than march
through a palette.

The bulb isn't supported by the Yeelight app. Developer mode was switched on
through python-miio, which opens port 55443; `start_music` then bypasses the
~60 commands/minute quota. Getting that far needs the bulb's miio token,
which lives in the Xiaomi cloud — see [Getting the token](#getting-the-token).

## Install

```bash
pip install yeelight numpy soundcard python-miio mss
pip install -r requirements.txt
```

The first line is the light engine. The second is the cloud/token half
(`requests`, `pycryptodome`, `Pillow`, `colorama`). If you already have a
token, you can skip it — the dashboard imports the extractor inside a
`try`/`except` and simply turns off the Xiaomi login in Setup when it isn't there.

macOS only, as written.

## Find the bulb first

Every command below wants `--ip`, and that address is whatever your own
router handed the bulb — there is no default, and nothing in this repo is
hard-coded to a particular network. Three ways to find it, best first:

1. **The desk's Setup page.** The device table has an **Address** column: the cloud
   already knows every device's LAN address, so the same login that fetches
   your token hands you the address to point `--ip` at. It is the only route
   that works *before* developer mode is on, and you have to do that login
   anyway — so in practice you never hunt for the address separately. See
   [Getting the token](#getting-the-token).
2. **SSDP discovery**, once developer mode is on:
   ```bash
   python -c "from yeelight import discover_bulbs; print(discover_bulbs())"
   ```
3. **Your router's DHCP lease table**, or `arp -a`, and match the MAC the
   cloud reported.

Give the bulb a DHCP reservation while you are in the router anyway. A lease
that expires and moves the bulb to a new address looks exactly like a dead
bulb from here.

The rest of this README writes that address as `BULB_IP`.

## Quick start

```bash
BULB_IP=192.0.2.10          # yours, not this one
```
```bash
python music_bulb_dash.py --ip "$BULB_IP"
```

Open the URL it prints. Everything is live: mode, palette, sensitivity,
capture rate, the lot. The desk can also take the address from its Setup
page, in which case you can start it with no `--ip` at all. Or drive it
headless:

```bash
python music_bulb.py --ip "$BULB_IP"                    # music
python music_bulb.py --ip "$BULB_IP" --screen           # screen
python music_bulb.py --ip "$BULB_IP" --fusion           # both
python music_bulb.py --ip "$BULB_IP" --fade 1.5         # sweep, don't snap
```

`--fade SECONDS` keeps the palette anchors and the onsets that pick them;
only the traversal between two anchors becomes a timed sweep on a smoothstep
ramp. Steps are spaced at least `fade` apart so one sweep lands before the
next begins.

## The desk

The stage across the top is the colour going to the bulb right now, with the
ring pulsing on beats and the strip below it showing 10 audio bands — or 12
hue bins in screen mode, each bar painted with the hue it counts. Underneath
sit the numbered panels: transport, palette, and whichever signal chain the
current mode uses.

The desk swaps its controls to match the mode. Fusion shows both halves minus
what it can't use: the palettes, the colour fade and the dark lift have
nothing to act on when the display owns the hue and the audio owns the
brightness.

Binds to localhost. `--host 0.0.0.0` reaches it from your phone — there is no
authentication, so only on a network you trust.

## Getting the token

The bulb's miio token is the one thing you can't read off the device. It
normally means a second checkout of
[Xiaomi-cloud-tokens-extractor](https://github.com/PiotrMachowski/Xiaomi-cloud-tokens-extractor).
That extractor is vendored here and wired into the desk instead, so the whole
setup is one repo:

1. Open **Setup**: click the bulb address in the top bar, or **Set up bulb**
   when none is set yet.
2. Log in with your Xiaomi account password, or scan a QR code in the Mi Home
   app. Captchas and 2FA codes are answered in the page.
3. The device table lists everything on the account: name, model, **Address**,
   token. Both halves of what you need are on that one row — the token to
   unlock the LAN API, and the address to talk to it on afterwards.
4. **Use** points the desk at that address, so you never have to type it.
   **Enable LAN** sends the token to the bulb, turns developer mode on and
   opens port 55443.

<img src="assets/desk-cloud.png" alt="the Setup page: bulb address field, password or QR login, region picker and a log pane" width="820">

That's the entire provisioning path, and it is also the easiest way to answer
"what is my bulb's IP?" — the cloud knows, and the table shows it. Afterwards
the desk never needs the cloud again; no token is stored anywhere in this
project.

Note the address the table reports is a DHCP lease, not a fixed property of
the bulb. If the light stops responding weeks later, check that row again
before assuming the bulb is dead.

The extractor still works as its original CLI:

```bash
python token_extractor.py
python token_extractor.py -ni -u you@example.com -p secret -s de -o out.json
```

### Handling of credentials

- Cloud endpoints are refused unless the server is bound to loopback, or you
  started it with `--allow-remote-login`. Your Xiaomi password crosses that
  socket and the desk has no authentication.
- The password is not kept after the login, and nothing is written to disk.
- Tokens are masked in the log pane; the device table reveals one row at a
  time, on request. They're account secrets — treat them like a password.

## Audio capture on macOS

`soundcard` has no CoreAudio loopback: `include_loopback=True` warns and is
ignored. BlackHole is a virtual cable, not a system tap — it only carries
what macOS *outputs* into it, so capturing from it while your output is the
built-in speakers gives you digital silence, which looks exactly like a
broken install.

To feed it, build a **Multi-Output Device** in Audio MIDI Setup containing the
physical output *and* BlackHole 2ch — physical device first as master, drift
correction ticked on BlackHole only, both members at the same sample rate —
then select it as the system output:

```bash
python music_bulb.py --ip "$BULB_IP" --mic "BlackHole 2ch"
python music_bulb.py --list-devices
```

With a Multi-Output device (or BlackHole itself) selected as the system
output, `--mic` is optional: the virtual cable is picked automatically, in the
CLI and on the desk. With a plain output selected it falls back to the
microphone.

Multi-Output devices ignore the hardware volume keys: set the level in the
playing app, and leave BlackHole's own gain at full or the capture goes quiet.

## Screen capture on macOS

Uses `mss`, about 20-26 ms for a grab of any size, so ~33 fps is the practical
ceiling. Most of that cost lands in macOS itself (WindowServer and the
permission daemons), not in this process, so the capture rate is the battery
dial: it defaults to 10 fps and drops to 5 and then 2 while the picture holds
still, returning to full on the first frame that changes. Without Screen Recording permission macOS hands back the desktop
wallpaper instead of an error, so the code preflights the permission and
refuses to start rather than quietly lighting your room the colour of your
wallpaper.

```bash
python music_bulb.py --list-monitors     # index 0 is the union of all displays
```

## When the bulb stops answering

**"Bulb closed the connection."** The bulb accepts only a handful of LAN
connections at once, and every dropped session used to burn a slot until it
timed out — a few failed starts could lock it out completely. `open_bulb()`
and `release_bulb()` now own that lifecycle: retry with backoff, force-close
the socket on every exit path. Nothing should construct `yeelight.Bulb`
directly.

If it's already wedged, that fix can't reach it — power-cycle at the switch.
To tell a wedged bulb from a dead one, ask the channels that don't use the
control socket:

```bash
python -c "from yeelight import discover_bulbs; print(discover_bulbs())"
```

SSDP keeps advertising power/bright/hue/sat, and a raw miio hello on UDP
54321 keeps answering, even while port 55443 accepts TCP and then hangs up.
Both alive plus 55443 refusing means burnt connection slots — not a network
or firmware problem.

And ping the bulb before blaming the code. Round trips here run 40–90 ms over
Wi-Fi, which is high for a LAN and shows up as lag at 25 updates/sec.

## Limits worth knowing

- Don't exceed ~30 updates/sec. The LEDs can't resolve past it, so both loops
  clamp the rate and drop frames whose HSV sits inside the perceptual
  deadband of the last one sent.
- In fusion mode the audio loop must call `record()` every ~23 ms or the input
  buffer backs up and onsets arrive late. A full-display grab costs ~26 ms, so
  capture runs on its own thread and publishes the latest colour; its scene-cut
  flag is sticky until read, so a cut can't fall into the gap between sends.

## Layout

```
music_bulb.py        DSP, screen capture, CLI, bulb lifecycle
music_bulb_dash.py   the web desk; CloudSession drives the login
token_extractor.py   Xiaomi cloud token extractor, importable and still a CLI
```

## Credits

The token extractor is
[Piotr Machowski's](https://github.com/PiotrMachowski/Xiaomi-cloud-tokens-extractor),
vendored here and refactored to be importable — `argparse` moved into
`main()`, and every `input()`/`print()` in the login flow put behind a
`Prompter` seam so the terminal and the browser can drive the same code. See
the upstream project for its licence and for the flows this doesn't exercise.

Bulb control via [yeelight](https://gitlab.com/stavros/python-yeelight) and
[python-miio](https://github.com/rytilahti/python-miio).
