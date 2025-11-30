# bone demo (multi-role binding)

Run a second demo to bind skeletons for multiple roles in `vision_resources` (`niu`, `sun`, `tu`) without changing the original demo.

## How to run

1. Install dependencies (same as `try/requirements.txt`):
```
pip install -r bone/requirements.txt
```
2. Start the demo:
```
python bone/demo.py
```

## Controls

- `1/2/3`: Switch role (`niu`/`sun`/`tu`).
- Arrow keys: Move selected part (dx/dy).
- `Q/E`: Rotate selected part (deg).
- `Z/X`: Scale selected part.
- `Tab`: Cycle selected part.
- `Space`: Toggle skeleton overlay.
- `S`: Save anchors immediately.
- Close window or press `Esc`: Auto-save current role anchors.

## Anchors persistence

- Each role stores anchors in `bone/anchors_<role>.json`.
- First-time alignment uses `__config.autoscale: true` to auto-scale from `scale: 1.0`.
- After any manual adjustment, autoscale is turned off and saved; subsequent runs keep your exact alignment.