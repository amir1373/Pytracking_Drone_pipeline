# PyTracking Drone Pipeline

Modified PyTracking-based drone vision pipeline with scripts for webcam/drone feed tracking and sending bounding-box output to an external client.

## Contents

- `pytracking/` - tracking framework code.
- `ltr/` - training/evaluation support code from the tracking stack.
- `drone_feed.py` - drone/video feed integration entry point.
- `tracker_webcam.py` - webcam tracking entry point.
- `INSTALL.md` and `INSTALL_win.md` - installation notes.
- `MODEL_ZOO.md` - model reference notes.

## Setup

Start with the installation guides in this repository:

- `INSTALL.md` for Linux-style setup.
- `INSTALL_win.md` for Windows setup.

The exact environment depends on the tracker/model being used and whether the input is a webcam, recorded stream, or drone video feed.

## Notes

This repository contains modified third-party tracking code. Check the license before reusing or redistributing derivative work.