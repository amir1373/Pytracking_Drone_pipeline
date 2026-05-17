# PyTracking Drone Pipeline

Modified PyTracking-based drone vision pipeline with scripts for webcam/drone feed tracking and sending bounding-box output to an external client.

## What This Repository Contains

- `pytracking/` - tracking framework code.
- `ltr/` - training/evaluation support code from the tracking stack.
- `drone_feed.py` - drone/video-feed integration entry point.
- `tracker_webcam.py` - webcam tracking entry point.
- `INSTALL.md` and `INSTALL_win.md` - installation notes.
- `MODEL_ZOO.md` - model reference notes.
- `.github/workflows/python-compile.yml` - lightweight Python compile check.

## Setup

Start with the installation guides included in the repository:

- `INSTALL.md` for Linux-style setup.
- `INSTALL_win.md` for Windows setup.

The exact environment depends on the tracker/model being used and whether the input is a webcam, recorded stream, or drone video feed.

## Project Notes

This repository contains modified third-party tracking code. Document any local changes to the original PyTracking workflow, especially the bounding-box UDP/client integration.

## License

Check the repository license before reusing or redistributing derivative work.