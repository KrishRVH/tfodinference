# TensorFlow Object Detection Inference

Oklahoma State University ECEN 4273 Project 2 by Krish Ravi and Jacobo Rosillo.

This project runs a TensorFlow Object Detection SavedModel against a video and
writes a video with bounding boxes.

## Repository Contents

- `src/tfodinference/`: command-line inference package.
- `labels/label_map.pbtxt`: label map for the trained detector.
- `docs/assets/`: project screenshots and final-deliverable images.
- `models/`: local exported SavedModel directories.
- `testvideos/`: local input videos.
- `resultvideos/`: generated inference videos.

The historical script referenced `testvideos/cursed.mp4` and
`testvideos/testpikachu.mp4`, but those videos are not present in the reachable
Git history, remote refs, or local dangling objects in this clone.

## Requirements

- `mise`
- Python 3.13.14
- `uv`

TensorFlow 2.21 publishes wheels for Python 3.13, but not Python 3.14, so this
repo intentionally uses Python 3.13 even though the shared Python standards
template has moved to Python 3.14.

## Setup

```sh
mise run install
```

Place your exported model at `models/BACKUP/saved_model` or pass `--model`.
Place input videos under `testvideos/` or pass the video path directly.

## Usage

```sh
mise run infer -- testvideos/cursed.mp4
```

The default output is `resultvideos/<video filename>.avi`, matching the original
project naming convention. Use `--output` for an exact path.

```sh
mise run infer -- \
  --model models/person/saved_model \
  --labels labels/label_map.pbtxt \
  --output resultvideos/person-demo.avi \
  testvideos/demo.mp4
```

Add `--preview` to show frames while inference runs. Press `q` to stop the
preview.

## Development

```sh
mise run fmt
mise run lint
mise run test
mise run standards:check
```

The standards gate checks formatting, Ruff, basedpyright, Bandit, pytest,
package build, dependency lock freshness, and dependency audit.
