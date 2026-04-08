#!/usr/bin/env python3
"""Stitch videos together from slow pace to fast pace.

The script scans a folder, removes exact duplicate files, computes an audio pace
score for each remaining video, orders the videos from slow to fast, and repeats
faster videos more times before concatenating them into one output file.

Usage:
    python stitch_videos_by_pace.py /path/to/folder --output stitched.mp4

Requirements:
    pip install opencv-python numpy moviepy
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import cv2
import numpy as np

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wmv",
    ".flv",
    ".webm",
    ".mpg",
    ".mpeg",
}

CHUNK_SIZE = 8192


def is_video_file(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def collect_video_files(folder):
    video_files = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            if is_video_file(path):
                video_files.append(path)
    return video_files


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_unique_files(files):
    grouped_by_hash = defaultdict(list)
    for path in files:
        grouped_by_hash[file_sha256(path)].append(path)

    unique_files = []
    duplicate_groups = []
    for group in grouped_by_hash.values():
        unique_files.append(group[0])
        if len(group) > 1:
            duplicate_groups.append(group)
    return unique_files, duplicate_groups


def get_video_metadata(path):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    duration = frame_count / fps if fps else 0.0
    return {
        "frames": frame_count,
        "fps": fps,
        "duration": duration,
        "resolution": (width, height),
    }


def compute_audio_pace_score(path, sample_rate=22050, max_duration=30.0):
    with VideoFileClip(path) as clip:
        audio = clip.audio
        if audio is None:
            return 0.0

        analysis_duration = min(max_duration, clip.duration or 0.0)
        if analysis_duration <= 0:
            return 0.0

        audio_array = audio.subclip(0, analysis_duration).to_soundarray(fps=sample_rate)

    if audio_array.size == 0:
        return 0.0

    if audio_array.ndim > 1:
        mono_audio = audio_array.mean(axis=1)
    else:
        mono_audio = audio_array

    mono_audio = mono_audio.astype(np.float32)
    window_size = max(256, int(sample_rate * 0.05))
    if len(mono_audio) < window_size:
        padded = np.pad(mono_audio, (0, window_size - len(mono_audio)))
        windows = padded.reshape(1, window_size)
    else:
        trimmed_length = len(mono_audio) - (len(mono_audio) % window_size)
        if trimmed_length == 0:
            trimmed_length = len(mono_audio)
        windows = mono_audio[:trimmed_length].reshape(-1, window_size)

    rms_values = np.sqrt(np.mean(np.square(windows), axis=1))
    transient_strength = float(np.mean(np.abs(np.diff(rms_values)))) if len(rms_values) > 1 else 0.0
    zero_crossings = np.abs(np.diff(np.signbit(mono_audio))).mean() if len(mono_audio) > 1 else 0.0
    loudness = float(rms_values.mean()) if len(rms_values) else 0.0
    # Give stronger emphasis to audio energy and dynamics when ordering by pace.
    return loudness + (transient_strength * 3.0) + float(zero_crossings)


def ensure_even(value):
    value = max(2, int(round(value)))
    if value % 2:
        value += 1
    return value


def get_output_size(video_plan, target_height):
    for item in video_plan:
        width, height = item["resolution"]
        if width > 0 and height > 0:
            output_height = ensure_even(target_height or height)
            output_width = ensure_even(width * (output_height / height))
            return output_width, output_height
    raise RuntimeError("No valid clip resolution available for stitched output.")


def ffmpeg_concatenate_files(file_paths, output_path):
    list_file = os.path.join(os.path.dirname(output_path), "ffmpeg_concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as listener:
        for path in file_paths:
            escaped_path = path.replace("'", "'\\''")
            listener.write(f"file '{escaped_path}'\n")

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c",
        "copy",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg concat failed: "
            + (result.stderr or result.stdout or "unknown error")
        )


def assign_repeat_count(score, duration, min_score, max_score, order_index=None, total_items=None):
    if duration >= 20.0:
        return 1

    if duration < 10.0 and order_index is not None and total_items is not None:
        if order_index >= total_items // 2:
            return 4

    if max_score <= min_score:
        return 2

    normalized = (score - min_score) / (max_score - min_score)
    if normalized < 0.40:
        return 1
    if normalized < 0.65:
        return 2
    if normalized < 0.85:
        return 3
    return 4


def build_stitch_plan(files):
    plan = []
    skipped_paths = []
    for path in files:
        try:
            # Try to open with MoviePy to ensure readable and nonzero duration
            with VideoFileClip(path) as test_clip:
                if not test_clip.duration or test_clip.duration <= 0.05:
                    skipped_paths.append((path, "zero or unreadable duration"))
                    continue
        except Exception as exc:
            skipped_paths.append((path, f"unreadable by MoviePy: {exc}"))
            continue

        try:
            metadata = get_video_metadata(path)
        except Exception as exc:
            skipped_paths.append((path, f"metadata error: {exc}"))
            continue

        try:
            pace_score = compute_audio_pace_score(path)
        except Exception as exc:
            skipped_paths.append((path, f"audio pace fallback to 0.0: {exc}"))
            pace_score = 0.0

        plan.append(
            {
                "path": path,
                "pace_score": pace_score,
                "duration": metadata.get("duration", 0.0),
                "fps": metadata.get("fps", 0.0),
                "resolution": metadata.get("resolution", (0, 0)),
            }
        )

    plan.sort(key=lambda item: (item["pace_score"], item["path"]))
    scores = [item["pace_score"] for item in plan]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    total_items = len(plan)
    for index, item in enumerate(plan):
        item["repeat_count"] = assign_repeat_count(
            item["pace_score"],
            item["duration"],
            min_score,
            max_score,
            order_index=index,
            total_items=total_items,
        )
    return plan, skipped_paths


def render_stitched_video(plan, output_path, target_height=720, output_fps=30.0, max_clip_seconds=0.0):
    if not plan:
        raise RuntimeError("No clips available for stitched output.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_width, output_height = get_output_size(plan, target_height)
    loaded_clips = []
    skipped_paths = []

    try:
        for item in plan:
            for repeat_index in range(item["repeat_count"]):
                source_clip = None
                processed_clip = None
                try:
                    source_clip = VideoFileClip(item["path"])
                    source_clip.get_frame(0)
                    clip_end = source_clip.duration or 0.0
                    if max_clip_seconds and max_clip_seconds > 0:
                        clip_end = min(clip_end, max_clip_seconds)
                    if clip_end <= 0.05:
                        raise RuntimeError("clip duration too short")

                    processed_clip = source_clip.subclip(0, clip_end)
                    # Resize to even width and height for H.264 compatibility
                    processed_clip = processed_clip.resize(newsize=(int(output_width), int(output_height)))
                    processed_clip = processed_clip.set_fps(output_fps)
                    loaded_clips.append(processed_clip)
                except Exception as exc:
                    skipped_paths.append((item["path"], repeat_index + 1, str(exc)))
                    if processed_clip is not None:
                        processed_clip.close()
                    if source_clip is not None:
                        source_clip.close()

        loaded_clips = [clip for clip in loaded_clips if clip.duration and clip.duration > 0.05]
        if not loaded_clips:
            raise RuntimeError("Stitched output was empty after loading the clips.")

        final_clip = None
        try:
            try:
                final_clip = concatenate_videoclips(loaded_clips, method="chain")
            except Exception as exc:
                try:
                    final_clip = concatenate_videoclips(loaded_clips, method="compose")
                except Exception as exc2:
                    raise RuntimeError(f"Failed to concatenate clips: {exc}; fallback compose failed: {exc2}")

            try:
                final_clip.write_videofile(
                    output_path,
                    fps=output_fps,
                    codec="libx264",
                    audio_codec="aac",
                    logger='bar',  # Show ffmpeg output
                    verbose=True,
                    threads=2,
                    preset='ultrafast',
                    ffmpeg_params=[
                        "-profile:v", "baseline",
                        "-level", "3.0",
                        "-pix_fmt", "yuv420p"
                    ],
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to write output video: {exc}")
            total_duration = final_clip.duration or 0.0
            final_clip.close()
            return total_duration, skipped_paths
        except Exception as exc:
            if final_clip is not None:
                final_clip.close()
            temp_paths = []
            with tempfile.TemporaryDirectory(prefix="stitched_temp_") as temp_dir:
                for index, clip in enumerate(loaded_clips):
                    temp_path = os.path.join(temp_dir, f"clip_{index:04d}.mp4")
                    clip.write_videofile(
                        temp_path,
                        fps=output_fps,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None,
                        verbose=False,
                        threads=2,
                        preset='ultrafast',
                        ffmpeg_params=[
                            "-profile:v", "baseline",
                            "-level", "3.0",
                            "-pix_fmt", "yuv420p"
                        ],
                    )
                    temp_paths.append(temp_path)
                ffmpeg_concatenate_files(temp_paths, output_path)
                with VideoFileClip(output_path) as output_clip:
                    total_duration = output_clip.duration or 0.0
            return total_duration, skipped_paths
    finally:
        for clip in loaded_clips:
            clip.close()


def write_report(output_path, duplicate_groups, plan, total_duration=None, skipped_paths=None, skipped_plan_paths=None):
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("Exact duplicate groups removed before stitching:\n")
        if duplicate_groups:
            for index, group in enumerate(duplicate_groups, start=1):
                output_file.write(f"\nGroup {index}:\n")
                for path in group:
                    output_file.write(f"{path}\n")
        else:
            output_file.write("No exact duplicates found.\n")

        output_file.write("\nStitch order (slow pace -> fast pace):\n")
        for index, item in enumerate(plan, start=1):
            output_file.write(
                "{}. [{:.4f}] x{} {}\n".format(
                    index,
                    item["pace_score"],
                    item["repeat_count"],
                    item["path"],
                )
            )

        output_file.write("\nRendered duration:\n")
        if total_duration is None:
            output_file.write("No output was rendered.\n")
        else:
            output_file.write(f"{total_duration:.2f}s\n")

        output_file.write("\nSkipped clips:\n")
        if skipped_paths:
            for path, repeat_index, reason in skipped_paths:
                output_file.write(f"repeat {repeat_index}: {path}\n")
                output_file.write(f"  {reason}\n")
        else:
            output_file.write("No clips were skipped.\n")

        output_file.write("\nSkipped or downgraded during plan build:\n")
        if skipped_plan_paths:
            for path, reason in skipped_plan_paths:
                output_file.write(f"{path}\n")
                output_file.write(f"  {reason}\n")
        else:
            output_file.write("No files needed fallback handling during plan build.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Stitch videos from slow pace to fast pace and repeat faster clips more often.")
    parser.add_argument("folder", help="Folder to scan for videos")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--report", help="Optional text report path")
    parser.add_argument("--clip-seconds", type=float, default=0.0, help="Optional maximum seconds to use from each clip; 0 means full clip")
    parser.add_argument("--height", type=int, default=720, help="Output height for the stitched video")
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS for the stitched video")
    parser.add_argument("--extensions", nargs="*", help="Additional video extensions to include")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.extensions:
        for extension in args.extensions:
            normalized = extension if extension.startswith(".") else f".{extension}"
            VIDEO_EXTENSIONS.add(normalized.lower())

    if not os.path.isdir(args.folder):
        print(f"Folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)

    if args.clip_seconds < 0 or args.height <= 0 or args.fps <= 0:
        print("--clip-seconds cannot be negative, and --height/--fps must be positive.", file=sys.stderr)
        sys.exit(1)

    files = collect_video_files(args.folder)
    if not files:
        print("No video files found in the given folder.")
        return

    print(f"Scanning {len(files)} video file(s) in: {args.folder}")

    unique_files, duplicate_groups = build_unique_files(files)
    if duplicate_groups:
        print("\nRemoved exact duplicates before stitching:")
        for index, group in enumerate(duplicate_groups, start=1):
            print(f"  Group {index}:")
            for path in group:
                print(f"    {path}")

    skipped_plan_paths = None
    try:
        plan, skipped_plan_paths = build_stitch_plan(unique_files)
    except Exception as exc:
        print(f"Failed to build stitch plan: {exc}", file=sys.stderr)
        sys.exit(1)

    if not plan:
        print("No stitchable clips remained after reading metadata and pace.", file=sys.stderr)
        sys.exit(1)

    print("\nStitch order (slow pace -> fast pace):")
    for index, item in enumerate(plan, start=1):
        print(
            "  {}. [{:.4f}] x{} {}".format(
                index,
                item["pace_score"],
                item["repeat_count"],
                item["path"],
            )
        )
    if skipped_plan_paths:
        print("Skipped or downgraded during plan build:")
        for path, reason in skipped_plan_paths:
            print(f"  {path}")
            print(f"    {reason}")

    total_duration = None
    skipped_paths = None
    try:
        total_duration, skipped_paths = render_stitched_video(
            plan,
            args.output,
            target_height=args.height,
            output_fps=args.fps,
            max_clip_seconds=args.clip_seconds,
        )
        print(f"\nStitched video written to: {args.output} ({total_duration:.2f}s)")
        if skipped_paths:
            print("Skipped clips:")
            for path, repeat_index, reason in skipped_paths:
                print(f"  repeat {repeat_index}: {path}")
                print(f"    {reason}")
    except Exception as exc:
        print(f"Failed to render stitched video: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.report:
        try:
            write_report(
                args.report,
                duplicate_groups,
                plan,
                total_duration=total_duration,
                skipped_paths=skipped_paths,
                skipped_plan_paths=skipped_plan_paths,
            )
            print(f"Report written to: {args.report}")
        except Exception as exc:
            print(f"Failed to write report: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()