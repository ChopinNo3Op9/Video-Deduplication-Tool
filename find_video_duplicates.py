#!/usr/bin/env python3
"""Find duplicate and similar videos in a folder and combine them.

Exact duplicates are detected by file hash. Similar videos are detected by sampling frames
and comparing perceptual image hashes. The optional combinator mode orders videos by
audio pace from slow to fast, trims overlapping visual sections between adjacent clips,
and writes one combined video with original audio.

Usage:
    python find_video_duplicates.py /path/to/folder

Requirements:
    pip install opencv-python numpy pillow imagehash moviepy
"""

import argparse
import hashlib
import os
import shutil
import sys
from collections import defaultdict

try:
    import cv2
    import numpy as np
    from PIL import Image
    import imagehash
    from moviepy import VideoFileClip, concatenate_videoclips
except ImportError as exc:
    print("Missing dependency: {}".format(exc), file=sys.stderr)
    print("Install required packages with: pip install opencv-python numpy pillow imagehash moviepy", file=sys.stderr)
    sys.exit(1)

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


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_video_metadata(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps else 0.0
    return {
        "frames": frame_count,
        "fps": fps,
        "duration": duration,
        "resolution": (width, height),
    }


def sample_frame_hashes(path, sample_count=8, hash_size=16):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []

    positions = [int(frame_count * i / (sample_count + 1)) for i in range(1, sample_count + 1)]
    hashes = []

    for frame_idx in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success or frame is None:
            continue
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        hashes.append(imagehash.phash(pil_image, hash_size=hash_size))

    cap.release()
    return hashes


def sample_frame_hashes_in_range(path, start_ratio=0.0, end_ratio=1.0, sample_count=8, hash_size=16):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []

    start_ratio = min(max(start_ratio, 0.0), 1.0)
    end_ratio = min(max(end_ratio, start_ratio), 1.0)
    start_frame = int(round((frame_count - 1) * start_ratio))
    end_frame = int(round((frame_count - 1) * end_ratio))
    if end_frame <= start_frame:
        end_frame = min(frame_count - 1, start_frame + max(1, sample_count - 1))

    if sample_count <= 1:
        positions = [start_frame]
    else:
        positions = np.linspace(start_frame, end_frame, num=sample_count)

    hashes = []
    for frame_idx in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(frame_idx)))
        success, frame = cap.read()
        if not success or frame is None:
            continue
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        hashes.append(imagehash.phash(pil_image, hash_size=hash_size))

    cap.release()
    return hashes


def hamming_distance(hash1, hash2):
    return (hash1 - hash2)


def signature_distance(sig1, sig2):
    if not sig1 or not sig2:
        return float("inf")
    length = min(len(sig1), len(sig2))
    distances = [hamming_distance(sig1[i], sig2[i]) for i in range(length)]
    return sum(distances) / len(distances)


def collect_video_files(folder):
    result = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            if is_video_file(path):
                result.append(path)
    return result


def build_duplicate_index(files):
    hash_groups = defaultdict(list)
    for path in files:
        file_hash = file_sha256(path)
        hash_groups[file_hash].append(path)

    unique_files = []
    duplicate_groups = []
    representative_map = {}

    for group in hash_groups.values():
        representative = group[0]
        unique_files.append(representative)
        for path in group:
            representative_map[path] = representative
        if len(group) > 1:
            duplicate_groups.append(group)

    return unique_files, duplicate_groups, representative_map


def find_exact_duplicates(files):
    return build_duplicate_index(files)[1]


def find_similar_videos(files, sample_count=8, hash_size=16, threshold=25):
    signatures = {}
    for path in files:
        try:
            signatures[path] = sample_frame_hashes(path, sample_count=sample_count, hash_size=hash_size)
        except Exception as exc:
            print(f"Warning: could not process {path}: {exc}", file=sys.stderr)
            signatures[path] = []

    visited = set()
    similar_pairs = []

    for i, path_a in enumerate(files):
        for path_b in files[i + 1 :]:
            if path_a == path_b:
                continue
            distance = signature_distance(signatures[path_a], signatures[path_b])
            if distance <= threshold:
                similar_pairs.append((path_a, path_b, round(distance, 2)))

    return similar_pairs


def compute_audio_pace_score(path, sample_rate=22050, max_duration=30.0):
    with VideoFileClip(path) as clip:
        audio = clip.audio
        if audio is None:
            return 0.0

        analysis_duration = min(max_duration, clip.duration or 0.0)
        if analysis_duration <= 0:
            return 0.0

        audio_array = audio.subclipped(0, analysis_duration).to_soundarray(fps=sample_rate)

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


def build_combine_plan(unique_files, max_clips=None):
    metadata_cache = {}
    for path in unique_files:
        try:
            metadata_cache[path] = get_video_metadata(path)
        except Exception as exc:
            print(f"Warning: could not read metadata for {path}: {exc}", file=sys.stderr)
            metadata_cache[path] = {"duration": 0.0, "fps": 0.0, "resolution": (0, 0)}

    combine_plan = []
    for path in unique_files:
        try:
            audio_pace = compute_audio_pace_score(path)
        except Exception as exc:
            print(f"Warning: could not score audio pace for {path}: {exc}", file=sys.stderr)
            audio_pace = 0.0

        metadata = metadata_cache[path]
        combine_plan.append(
            {
                "path": path,
                "audio_pace": audio_pace,
                "duration": metadata.get("duration", 0.0),
                "fps": metadata.get("fps", 0.0),
                "resolution": metadata.get("resolution", (0, 0)),
            }
        )

    combine_plan.sort(key=lambda item: (item["audio_pace"], item["path"]))
    if max_clips and max_clips > 0:
        combine_plan = combine_plan[:max_clips]
    return combine_plan


def ensure_even(value):
    value = max(2, int(round(value)))
    if value % 2:
        value += 1
    return value


def get_output_size(combine_plan, target_height):
    for item in combine_plan:
        width, height = item["resolution"]
        if width > 0 and height > 0:
            output_height = ensure_even(target_height or height)
            output_width = ensure_even(width * (output_height / height))
            return (output_width, output_height)
    raise RuntimeError("No valid clip resolution available for combined output.")


def estimate_visual_overlap(previous_path, current_path, max_overlap_seconds=1.5, sample_count=12, hash_size=8, match_threshold=12.0):
    previous_metadata = get_video_metadata(previous_path)
    current_metadata = get_video_metadata(current_path)
    previous_duration = previous_metadata.get("duration", 0.0)
    current_duration = current_metadata.get("duration", 0.0)
    if previous_duration <= 0 or current_duration <= 0:
        return 0.0

    max_overlap = min(max_overlap_seconds, previous_duration * 0.5, current_duration * 0.5)
    if max_overlap <= 0.1:
        return 0.0

    best_overlap = 0.0
    candidate_steps = 6
    for step in range(1, candidate_steps + 1):
        candidate_overlap = max_overlap * (step / candidate_steps)
        previous_start_ratio = max(0.0, 1.0 - (candidate_overlap / previous_duration))
        current_end_ratio = min(1.0, candidate_overlap / current_duration)

        previous_hashes = sample_frame_hashes_in_range(
            previous_path,
            start_ratio=previous_start_ratio,
            end_ratio=1.0,
            sample_count=sample_count,
            hash_size=hash_size,
        )
        current_hashes = sample_frame_hashes_in_range(
            current_path,
            start_ratio=0.0,
            end_ratio=current_end_ratio,
            sample_count=sample_count,
            hash_size=hash_size,
        )

        distance = signature_distance(previous_hashes, current_hashes)
        if distance <= match_threshold:
            best_overlap = candidate_overlap

    return round(best_overlap, 3)


def build_overlap_signature_index(files, max_overlap_seconds=10.0, sample_count=12, hash_size=8):
    signature_index = {}

    for path in files:
        try:
            metadata = get_video_metadata(path)
            duration = metadata.get("duration", 0.0)
            if duration <= 0:
                signature_index[path] = {"head": [], "tail": []}
                continue

            head_end_ratio = min(1.0, max_overlap_seconds / duration)
            tail_start_ratio = max(0.0, 1.0 - (max_overlap_seconds / duration))
            signature_index[path] = {
                "head": sample_frame_hashes_in_range(
                    path,
                    start_ratio=0.0,
                    end_ratio=head_end_ratio,
                    sample_count=sample_count,
                    hash_size=hash_size,
                ),
                "tail": sample_frame_hashes_in_range(
                    path,
                    start_ratio=tail_start_ratio,
                    end_ratio=1.0,
                    sample_count=sample_count,
                    hash_size=hash_size,
                ),
            }
        except Exception as exc:
            print(f"Warning: could not precompute overlap signatures for {path}: {exc}", file=sys.stderr)
            signature_index[path] = {"head": [], "tail": []}

    return signature_index


def render_combined_video(combine_plan, output_path, target_height=720, output_fps=30.0, max_clip_seconds=None, max_overlap_seconds=1.5):
    if not combine_plan:
        raise RuntimeError("No clips available for combined output.")

    output_size = get_output_size(combine_plan, target_height)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    clips = []
    overlap_details = []
    previous_path = None
    skipped_paths = []

    try:
        for item in combine_plan:
            trim_start = 0.0
            if previous_path is not None:
                trim_start = estimate_visual_overlap(previous_path, item["path"], max_overlap_seconds=max_overlap_seconds)
            source_clip = None
            processed_clip = None
            try:
                source_clip = VideoFileClip(item["path"])
                source_clip.get_frame(0)
                effective_end = source_clip.duration
                if max_clip_seconds and max_clip_seconds > 0:
                    effective_end = min(effective_end, max_clip_seconds)

                if trim_start >= effective_end:
                    trim_start = max(0.0, effective_end - 0.05)

                processed_clip = source_clip.subclipped(trim_start, effective_end)
                processed_clip = processed_clip.resized(height=output_size[1])
                processed_clip = processed_clip.with_fps(output_fps)
                clips.append(processed_clip)
                overlap_details.append((item["path"], trim_start))
                previous_path = item["path"]
            except Exception as exc:
                skipped_paths.append((item["path"], str(exc)))
                print(f"Warning: skipping unreadable clip {item['path']}: {exc}", file=sys.stderr)
                if processed_clip is not None:
                    processed_clip.close()
                if source_clip is not None:
                    source_clip.close()

        clips = [clip for clip in clips if clip.duration and clip.duration > 0.05]
        if not clips:
            raise RuntimeError("Combined output was empty after trimming overlaps.")

        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(
            output_path,
            fps=output_fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        total_duration = final_clip.duration or 0.0
        final_clip.close()
        return total_duration, overlap_details, skipped_paths
    finally:
        for clip in clips:
            clip.close()


def build_trim_export_plan(files, max_overlap_seconds=10.0):
    export_plan = []
    signature_index = build_overlap_signature_index(files, max_overlap_seconds=max_overlap_seconds)

    for index, path in enumerate(files):
        best_overlap = 0.0
        overlap_source = None
        current_head = signature_index.get(path, {}).get("head", [])

        for previous_path in files[:index]:
            previous_tail = signature_index.get(previous_path, {}).get("tail", [])
            coarse_distance = signature_distance(previous_tail, current_head)
            if coarse_distance == float("inf") or coarse_distance > 14.0:
                continue

            try:
                overlap_seconds = estimate_visual_overlap(
                    previous_path,
                    path,
                    max_overlap_seconds=max_overlap_seconds,
                )
            except Exception as exc:
                print(f"Warning: could not compare overlap for {previous_path} -> {path}: {exc}", file=sys.stderr)
                continue

            if overlap_seconds > best_overlap:
                best_overlap = overlap_seconds
                overlap_source = previous_path

        export_plan.append(
            {
                "path": path,
                "trim_start": best_overlap,
                "overlap_source": overlap_source,
            }
        )

    return export_plan


def sanitize_output_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    safe_chars = []
    for char in stem:
        if char.isalnum() or char in {"-", "_", " ", "(", ")"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip() or "video"


def export_trimmed_videos(export_plan, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    exported_files = []
    skipped_paths = []

    for index, item in enumerate(export_plan, start=1):
        source_path = item["path"]
        trim_start = item.get("trim_start", 0.0) or 0.0
        output_stem = f"{index:03d}_{sanitize_output_stem(source_path)}"
        source_extension = os.path.splitext(source_path)[1].lower() or ".mp4"
        output_path = os.path.join(output_dir, output_stem + source_extension)

        source_clip = None
        processed_clip = None
        try:
            source_clip = VideoFileClip(source_path)
            source_clip.get_frame(0)
            duration = source_clip.duration or 0.0
            if duration <= 0.05:
                raise RuntimeError("clip duration too short")

            if trim_start >= duration:
                trim_start = max(0.0, duration - 0.05)

            if trim_start <= 0.001:
                shutil.copy2(source_path, output_path)
            else:
                output_path = os.path.join(output_dir, output_stem + ".mp4")
                processed_clip = source_clip.subclipped(trim_start, duration)
                if not processed_clip.duration or processed_clip.duration <= 0.05:
                    raise RuntimeError("trim removed the entire clip")
                processed_clip.write_videofile(
                    output_path,
                    fps=source_clip.fps or 30.0,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                )

            exported_files.append((source_path, output_path, trim_start, item.get("overlap_source")))
        except Exception as exc:
            skipped_paths.append((source_path, str(exc)))
            print(f"Warning: skipping export for {source_path}: {exc}", file=sys.stderr)
        finally:
            if processed_clip is not None:
                processed_clip.close()
            if source_clip is not None:
                source_clip.close()

    return exported_files, skipped_paths


def print_exact_duplicates(duplicate_groups):
    if not duplicate_groups:
        print("No exact duplicate videos found.")
        return

    print("Exact duplicate video groups:")
    for idx, group in enumerate(duplicate_groups, start=1):
        print(f"\nGroup {idx}:")
        for path in group:
            print(f"  {path}")


def print_similar_videos(similar_pairs):
    if similar_pairs is None:
        print("Similarity scan skipped.")
        return
    if not similar_pairs:
        print("No similar videos found.")
        return

    print("Similar video pairs:")
    for path_a, path_b, distance in similar_pairs:
        print(f"  [{distance}] {path_a}")
        print(f"       {path_b}")


def print_combine_plan(combine_plan):
    if not combine_plan:
        print("No combinator candidates found.")
        return

    print("Combinator plan (audio pace slow -> fast):")
    for index, item in enumerate(combine_plan, start=1):
        print(
            "  {}. [{:.4f}] {}".format(
                index,
                item["audio_pace"],
                item["path"],
            )
        )


def print_export_plan(export_plan):
    if not export_plan:
        print("No trimmed export plan generated.")
        return

    print("Trimmed folder export plan:")
    for index, item in enumerate(export_plan, start=1):
        overlap_source = item.get("overlap_source") or "no overlap match"
        print(
            "  {}. [{:.3f}s] {}".format(
                index,
                item.get("trim_start", 0.0),
                item["path"],
            )
        )
        if item.get("overlap_source"):
            print(f"       matched with: {overlap_source}")


def parse_args():
    parser = argparse.ArgumentParser(description="Find duplicated and similar videos in a folder.")
    parser.add_argument("folder", help="Folder to scan for videos")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of frames to sample per video")
    parser.add_argument("--hash-size", type=int, default=16, help="Perceptual hash size for frames")
    parser.add_argument("--threshold", type=float, default=25.0, help="Similarity threshold for video pairs")
    parser.add_argument("--extensions", nargs="*", help="Additional video file extensions to include")
    parser.add_argument("--output", "-o", help="Write results to a text file")
    parser.add_argument("--create-edit", help="Write one combined video to this path")
    parser.add_argument("--export-trimmed-folder", help="Write trimmed per-video outputs to this folder")
    parser.add_argument("--max-edit-clips", type=int, default=0, help="Optional maximum number of videos to combine; 0 means all unique videos")
    parser.add_argument("--clip-seconds", type=float, default=0.0, help="Optional maximum seconds to use from each input video; 0 means full video")
    parser.add_argument("--overlap-seconds", type=float, default=1.5, help="Maximum overlapping duration to detect and trim from the start of the next video")
    parser.add_argument("--edit-height", type=int, default=720, help="Output height for the combined video")
    parser.add_argument("--edit-fps", type=float, default=30.0, help="Output FPS for the combined video")
    return parser.parse_args()


def write_report(
    output_path,
    duplicate_groups,
    similar_pairs,
    combine_plan=None,
    overlap_details=None,
    skipped_clips=None,
    export_plan=None,
    exported_files=None,
    skipped_exports=None,
):
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("Exact duplicate video groups:\n")
        if duplicate_groups:
            for idx, group in enumerate(duplicate_groups, start=1):
                out.write(f"\nGroup {idx}:\n")
                for path in group:
                    out.write(f"{path}\n")
        else:
            out.write("No exact duplicate videos found.\n")

        out.write("\nSimilar video pairs:\n")
        if similar_pairs is None:
            out.write("Similarity scan skipped.\n")
        elif similar_pairs:
            for path_a, path_b, distance in similar_pairs:
                out.write(f"[{distance}] {path_a}\n")
                out.write(f"{path_b}\n\n")
        else:
            out.write("No similar videos found.\n")

        out.write("\nCombinator plan (audio pace slow -> fast):\n")
        if combine_plan:
            for index, item in enumerate(combine_plan, start=1):
                out.write(
                    "{}. [{:.4f}] {}\n".format(
                        index,
                        item["audio_pace"],
                        item["path"],
                    )
                )
        else:
            out.write("No combinator plan generated.\n")

        out.write("\nTrimmed overlap at clip starts:\n")
        if overlap_details:
            for path, trim_start in overlap_details:
                out.write(f"{trim_start:.3f}s {path}\n")
        else:
            out.write("No overlap trimming details available.\n")

        out.write("\nSkipped unreadable clips:\n")
        if skipped_clips:
            for path, reason in skipped_clips:
                out.write(f"{path}\n")
                out.write(f"  {reason}\n")
        else:
            out.write("No clips were skipped.\n")

        out.write("\nTrimmed folder export plan:\n")
        if export_plan:
            for item in export_plan:
                out.write(f"{item.get('trim_start', 0.0):.3f}s {item['path']}\n")
                if item.get("overlap_source"):
                    out.write(f"  matched with: {item['overlap_source']}\n")
        else:
            out.write("No trimmed folder export plan generated.\n")

        out.write("\nExported trimmed files:\n")
        if exported_files:
            for source_path, output_path_value, trim_start, overlap_source in exported_files:
                out.write(f"{trim_start:.3f}s {source_path}\n")
                out.write(f"  output: {output_path_value}\n")
                if overlap_source:
                    out.write(f"  matched with: {overlap_source}\n")
        else:
            out.write("No trimmed videos were exported.\n")

        out.write("\nSkipped trimmed exports:\n")
        if skipped_exports:
            for path, reason in skipped_exports:
                out.write(f"{path}\n")
                out.write(f"  {reason}\n")
        else:
            out.write("No trimmed exports were skipped.\n")


def main():
    args = parse_args()

    if args.extensions:
        for ext in args.extensions:
            normalized = ext if ext.startswith(".") else f".{ext}"
            VIDEO_EXTENSIONS.add(normalized.lower())

    if not os.path.isdir(args.folder):
        print(f"Folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)

    if args.max_edit_clips < 0:
        print("--max-edit-clips cannot be negative.", file=sys.stderr)
        sys.exit(1)
    if args.clip_seconds < 0 or args.overlap_seconds < 0 or args.edit_height <= 0 or args.edit_fps <= 0:
        print("Video combine settings must be positive values, and --clip-seconds/--overlap-seconds cannot be negative.", file=sys.stderr)
        sys.exit(1)

    files = collect_video_files(args.folder)
    if not files:
        print("No video files found in the given folder.")
        return

    print(f"Scanning {len(files)} video file(s) in: {args.folder}")

    print("\nChecking exact duplicates...")
    unique_files, duplicate_groups, representative_map = build_duplicate_index(files)
    print_exact_duplicates(duplicate_groups)

    should_scan_similarity = args.create_edit or not args.export_trimmed_folder

    print("\nChecking visually similar videos...")
    if should_scan_similarity:
        similar_pairs = find_similar_videos(
            unique_files,
            sample_count=args.sample_count,
            hash_size=args.hash_size,
            threshold=args.threshold,
        )
    else:
        similar_pairs = None
    print_similar_videos(similar_pairs)

    combine_plan = None
    overlap_details = None
    skipped_clips = None
    export_plan = None
    exported_files = None
    skipped_exports = None
    if args.create_edit:
        print("\nBuilding combinator plan...")
        combine_plan = build_combine_plan(
            unique_files,
            max_clips=args.max_edit_clips if args.max_edit_clips else None,
        )
        print_combine_plan(combine_plan)

        if combine_plan:
            try:
                total_duration, overlap_details, skipped_clips = render_combined_video(
                    combine_plan,
                    args.create_edit,
                    target_height=args.edit_height,
                    output_fps=args.edit_fps,
                    max_clip_seconds=args.clip_seconds if args.clip_seconds else None,
                    max_overlap_seconds=args.overlap_seconds,
                )
                print(f"\nCombined video written to: {args.create_edit} ({total_duration:.2f}s)")
                print("Trimmed overlap at clip starts:")
                for path, trim_start in overlap_details:
                    print(f"  {trim_start:.3f}s {path}")
                if skipped_clips:
                    print("Skipped unreadable clips:")
                    for path, reason in skipped_clips:
                        print(f"  {path}")
                        print(f"    {reason}")
            except Exception as exc:
                print(f"Failed to create combined video: {exc}", file=sys.stderr)

    if args.export_trimmed_folder:
        print("\nBuilding trimmed folder export plan...")
        export_plan = build_trim_export_plan(files, max_overlap_seconds=args.overlap_seconds)
        print_export_plan(export_plan)

        try:
            exported_files, skipped_exports = export_trimmed_videos(export_plan, args.export_trimmed_folder)
            print(f"\nTrimmed videos written to: {args.export_trimmed_folder}")
            for source_path, output_path_value, trim_start, overlap_source in exported_files:
                print(f"  [{trim_start:.3f}s] {source_path}")
                print(f"       {output_path_value}")
                if overlap_source:
                    print(f"       matched with: {overlap_source}")
            if skipped_exports:
                print("Skipped trimmed exports:")
                for path, reason in skipped_exports:
                    print(f"  {path}")
                    print(f"    {reason}")
        except Exception as exc:
            print(f"Failed to export trimmed folder: {exc}", file=sys.stderr)

    if args.output:
        try:
            write_report(
                args.output,
                duplicate_groups,
                similar_pairs,
                combine_plan=combine_plan,
                overlap_details=overlap_details,
                skipped_clips=skipped_clips,
                export_plan=export_plan,
                exported_files=exported_files,
                skipped_exports=skipped_exports,
            )
            print(f"\nResults written to: {args.output}")
        except Exception as exc:
            print(f"Failed to write output file: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
