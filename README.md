# Shadow Puppet Interactive — Sun Wukong and the Banana Fan

## Overview
- Category: Interactive experience inspired by galgame
- Art Style: Chinese traditional shadow puppetry
- Concept: This project reimagines Journey to the West through multi-perspective storytelling and tactile performance. It adapts the classic "Sun Wukong Borrows the Banana Fan Three Times" into an interactive experience with shadow-puppet aesthetics.
- Narrative: Uses three-line storytelling, shifting perspectives among monsters, the heavenly court, and other roles, allowing players to reinterpret the classic through fresh viewpoints while appreciating the artistry of shadow puppetry.

## Core Gameplay
- Players use camera-based hand gesture recognition to select and directly control shadow-puppet characters, performing set pieces and poses to progress.
- Combines traditional visual aesthetics with motion-driven interaction to deliver an immersive cultural and entertainment experience.

## Project Structure
- Developer folders: `bone/`, `collision_volume/`, `pose/`
  - Visual tools to bind puppet bones, attach collision volumes, and define puppet poses.
- Assets: `Audio/`, `video/`, `vision_resources/`
- Entrypoints: `open/`, `main_menu/`, `handpose2/`
  - Contains direct scene launchers and demos.

## Environment Setup
- Install Python 3 and ensure `python` and `pip` work in your terminal.
- Create a virtual environment at the project root:

```
python -m venv .venv
```

- Activate it (Windows PowerShell):

```
\.venv\Scripts\Activate.ps1
```

- Or activate it (Windows cmd):

```
\.venv\Scripts\activate.bat
```

- Upgrade `pip` and install dependencies:

```
python -m pip install --upgrade pip
pip install -r handpose2/requirements.txt
pip install -r bone/requirements.txt
```

- Deactivate when done:

```
deactivate
```

## How to Run
- Full flow: start from `open/hand_gate.py` and follow the on-screen guidance through the experience.
- Story mode: run `handpose2/story1.py` to jump directly into the story.
