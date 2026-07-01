"""Run TensorFlow Object Detection inference over a video file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

CODEC_LENGTH: Final = 4
DEFAULT_CODEC: Final = "MJPG"
DEFAULT_FPS: Final = 10.0
DEFAULT_LABELS: Final = Path("labels/label_map.pbtxt")
DEFAULT_MODEL: Final = Path("models/BACKUP/saved_model")
DEFAULT_OUTPUT_DIR: Final = Path("resultvideos")
DEFAULT_VIDEO: Final = Path("testvideos/cursed.mp4")
MASK_THRESHOLD: Final = 0.5
BOX_LINE_THICKNESS: Final = 8
QUIT_KEY: Final = ord("q")


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Resolved settings for one video inference run."""

    codec: str
    fps: float
    labels: Path
    model: Path
    output: Path
    preview: bool
    video: Path


@dataclass(frozen=True, slots=True)
class Runtime:
    """Runtime modules loaded lazily because TensorFlow imports are expensive."""

    category_index: dict[int, dict[str, int | str]]
    cv2: Any
    np: Any
    tf: Any
    utils_ops: Any
    vis_util: Any


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="tfodinference",
        description="Run TensorFlow Object Detection inference on a video.",
    )
    parser.add_argument(
        "video",
        default=DEFAULT_VIDEO,
        nargs="?",
        type=Path,
        help=f"video to process (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        type=Path,
        help=f"exported SavedModel directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        type=Path,
        help=f"TensorFlow label map (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"directory for generated videos (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="exact output path; overrides --output-dir",
    )
    parser.add_argument(
        "--fps",
        default=DEFAULT_FPS,
        type=float,
        help=f"output frames per second (default: {DEFAULT_FPS:g})",
    )
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        help=f"four-character OpenCV codec (default: {DEFAULT_CODEC})",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show frames while inference runs; press q to stop",
    )
    return parser


def default_output_path(video: Path, output_dir: Path) -> Path:
    """Return the legacy output filename for a source video."""
    return output_dir / f"{video.name}.avi"


def resolve_config(args: argparse.Namespace, *, cwd: Path | None = None) -> InferenceConfig:
    """Resolve parser output into absolute inference paths."""
    base = Path.cwd() if cwd is None else cwd
    video = _resolve_path(cast("Path", args.video), base)
    output_dir = _resolve_path(cast("Path", args.output_dir), base)
    output_arg = cast("Path | None", args.output)
    output = (
        _resolve_path(output_arg, base) if output_arg else default_output_path(video, output_dir)
    )

    return InferenceConfig(
        codec=cast("str", args.codec),
        fps=cast("float", args.fps),
        labels=_resolve_path(cast("Path", args.labels), base),
        model=_resolve_path(cast("Path", args.model), base),
        output=output,
        preview=cast("bool", args.preview),
        video=video,
    )


def run(config: InferenceConfig) -> Path:
    """Run object detection for the configured video and return the output path."""
    _validate_config(config)
    runtime = _load_runtime(config.labels)
    model = runtime.tf.saved_model.load(str(config.model))
    _write_video(config, runtime, model)
    return config.output


def run_inference_for_single_image(runtime: Runtime, model: Any, image: Any) -> dict[str, Any]:
    """Run TensorFlow inference for one RGB image."""
    image_array = runtime.np.asarray(image)
    input_tensor = runtime.tf.convert_to_tensor(image_array)
    input_tensor = input_tensor[runtime.tf.newaxis, ...]

    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(runtime.np.int64)

    if "detection_masks" in detections:
        detection_masks_reframed = runtime.utils_ops.reframe_box_masks_to_image_masks(
            detections["detection_masks"],
            detections["detection_boxes"],
            image_array.shape[0],
            image_array.shape[1],
        )
        detection_masks_reframed = runtime.tf.cast(
            detection_masks_reframed > MASK_THRESHOLD,
            runtime.tf.uint8,
        )
        detections["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return detections


def render_inference(runtime: Runtime, model: Any, image: Any) -> Any:
    """Draw detection boxes and labels onto an RGB image."""
    detections = run_inference_for_single_image(runtime, model, image)
    return runtime.vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        detections["detection_boxes"],
        detections["detection_classes"],
        detections["detection_scores"],
        runtime.category_index,
        instance_masks=detections.get("detection_masks_reframed"),
        line_thickness=BOX_LINE_THICKNESS,
        use_normalized_coordinates=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(resolve_config(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(2, f"tfodinference: {error}\n")
    return 0


def _load_runtime(labels: Path) -> Runtime:
    import cv2
    import numpy as np
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import ops as utils_ops
    from object_detection.utils import visualization_utils as vis_util

    utils_ops.tf = tf.compat.v1
    tf_compat = cast("Any", tf)
    tf_compat.gfile = tf.io.gfile
    category_index = label_map_util.create_category_index_from_labelmap(
        str(labels),
        use_display_name=True,
    )
    return Runtime(
        category_index=category_index,
        cv2=cv2,
        np=np,
        tf=tf,
        utils_ops=utils_ops,
        vis_util=vis_util,
    )


def _resolve_path(path: Path, cwd: Path) -> Path:
    return path if path.is_absolute() else cwd / path


def _validate_config(config: InferenceConfig) -> None:
    if len(config.codec) != CODEC_LENGTH:
        msg = f"codec must be four characters, got {config.codec!r}"
        raise ValueError(msg)

    required_paths = {
        "labels": config.labels,
        "model": config.model,
        "video": config.video,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        msg = "missing required input paths:\n" + "\n".join(missing)
        raise FileNotFoundError(msg)

    if not config.model.is_dir():
        msg = f"model must be an exported SavedModel directory: {config.model}"
        raise ValueError(msg)

    if config.fps <= 0:
        msg = f"fps must be greater than zero, got {config.fps:g}"
        raise ValueError(msg)

    config.output.parent.mkdir(parents=True, exist_ok=True)


def _write_video(config: InferenceConfig, runtime: Runtime, model: Any) -> None:
    cap = runtime.cv2.VideoCapture(str(config.video))
    try:
        if not cap.isOpened():
            msg = f"could not open video: {config.video}"
            raise RuntimeError(msg)

        frame_width = int(cap.get(runtime.cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(runtime.cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width <= 0 or frame_height <= 0:
            msg = f"could not read video dimensions: {config.video}"
            raise RuntimeError(msg)

        writer = runtime.cv2.VideoWriter(
            str(config.output),
            runtime.cv2.VideoWriter_fourcc(*config.codec),
            config.fps,
            (frame_width, frame_height),
        )
        try:
            if not writer.isOpened():
                msg = f"could not open output video: {config.output}"
                raise RuntimeError(msg)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb_frame = runtime.cv2.cvtColor(frame, runtime.cv2.COLOR_BGR2RGB)
                rendered_rgb = render_inference(runtime, model, rgb_frame)
                rendered_bgr = runtime.cv2.cvtColor(rendered_rgb, runtime.cv2.COLOR_RGB2BGR)
                writer.write(rendered_bgr)

                if config.preview:
                    runtime.cv2.imshow("tfodinference", rendered_bgr)
                    if runtime.cv2.waitKey(1) == QUIT_KEY:
                        break
        finally:
            writer.release()
    finally:
        cap.release()
        if config.preview:
            runtime.cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
