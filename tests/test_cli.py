"""Tests for the tfodinference command-line configuration."""

from pathlib import Path

from tfodinference.cli import build_parser, default_output_path, resolve_config


def test_default_config_uses_project_input_locations() -> None:
    """Default arguments point at the historical project paths."""
    config = resolve_config(build_parser().parse_args([]), cwd=Path("/repo"))

    assert config.labels == Path("/repo/labels/label_map.pbtxt")
    assert config.model == Path("/repo/models/BACKUP/saved_model")
    assert config.output == Path("/repo/resultvideos/cursed.mp4.avi")
    assert config.video == Path("/repo/testvideos/cursed.mp4")


def test_output_path_preserves_legacy_video_suffix() -> None:
    """The default output keeps the original video filename before .avi."""
    assert default_output_path(
        Path("/repo/testvideos/testpikachu.mp4"),
        Path("/repo/resultvideos"),
    ) == Path("/repo/resultvideos/testpikachu.mp4.avi")


def test_explicit_output_overrides_output_dir() -> None:
    """An explicit output path wins over the generated default."""
    parser = build_parser()
    config = resolve_config(
        parser.parse_args(
            [
                "--output-dir",
                "ignored",
                "--output",
                "custom/out.avi",
                "clips/input.mp4",
            ],
        ),
        cwd=Path("/repo"),
    )

    assert config.output == Path("/repo/custom/out.avi")
