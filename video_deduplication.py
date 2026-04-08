#!/usr/bin/env python3
"""Find exact duplicate and visually similar videos in a folder.

This is the lightweight scanning-only entrypoint split out from the broader
video combine/export workflow.

Usage:
    python video_deduplication.py /path/to/folder

Requirements:
    pip install opencv-python pillow imagehash
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
    from moviepy.editor import VideoFileClip
except ImportError as exc:
    print(f"Missing dependency: {exc}", file=sys.stderr)
    print("Install required packages with: pip install opencv-python pillow imagehash numpy moviepy", file=sys.stderr)
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
    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def sample_frame_hashes(path, sample_count=8, hash_size=16):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        return []

    positions = [int(frame_count * index / (sample_count + 1)) for index in range(1, sample_count + 1)]
    hashes = []

    for frame_idx in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        if not success or frame is None:
            continue
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hashes.append(imagehash.phash(Image.fromarray(image), hash_size=hash_size))

    capture.release()
    return hashes


def signature_distance(signature_a, signature_b):
    if not signature_a or not signature_b:
        return float("inf")

    length = min(len(signature_a), len(signature_b))
    distances = [(signature_a[index] - signature_b[index]) for index in range(length)]
    return sum(distances) / len(distances)


def numeric_signature_distance(signature_a, signature_b):
    if signature_a is None or signature_b is None:
        return float("inf")
    if len(signature_a) == 0 or len(signature_b) == 0:
        return float("inf")

    array_a = np.asarray(signature_a, dtype=np.float32)
    array_b = np.asarray(signature_b, dtype=np.float32)
    length = min(len(array_a), len(array_b))
    if length == 0:
        return float("inf")
    array_a = array_a[:length]
    array_b = array_b[:length]
    return float(np.mean(np.abs(array_a - array_b)))


def collect_video_files(folder):
    video_files = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            if is_video_file(path):
                video_files.append(path)
    return video_files


def sample_audio_signature(path, sample_rate=8000, signature_length=64, max_duration=30.0):
    try:
        with VideoFileClip(path) as clip:
            audio = clip.audio
            if audio is None:
                return None

            analysis_duration = min(max_duration, clip.duration or 0.0)
            if analysis_duration <= 0:
                return None

            def try_audio_duration(duration):
                try:
                    return audio.subclip(0, duration).to_soundarray(fps=sample_rate)
                except Exception:
                    return None

            sound_array = try_audio_duration(analysis_duration)
            if sound_array is None and analysis_duration > 1.0:
                for duration in (min(10.0, analysis_duration), min(5.0, analysis_duration), min(2.0, analysis_duration), 1.0):
                    sound_array = try_audio_duration(duration)
                    if sound_array is not None:
                        break

        if not isinstance(sound_array, np.ndarray) or sound_array.size == 0:
            return None

        if sound_array.ndim > 1:
            mono_audio = sound_array.mean(axis=1)
        else:
            mono_audio = sound_array

        mono_audio = mono_audio.astype(np.float32)
        if len(mono_audio) < 2:
            return None

        peak = float(np.max(np.abs(mono_audio)))
        if peak > 0:
            mono_audio = mono_audio / peak

        if len(mono_audio) < signature_length:
            mono_audio = np.pad(mono_audio, (0, signature_length - len(mono_audio)), mode='constant')

        source_indexes = np.linspace(0, len(mono_audio) - 1, num=len(mono_audio), dtype=np.float32)
        target_indexes = np.linspace(0, len(mono_audio) - 1, num=signature_length, dtype=np.float32)
        sampled_waveform = np.interp(target_indexes, source_indexes, mono_audio)
        envelope = np.abs(sampled_waveform)
        if sampled_waveform.shape != envelope.shape:
            return None
        return np.concatenate([sampled_waveform, envelope]).tolist()
    except Exception as exc:
        print(f"Warning: error extracting audio signature for {path}: {exc}", file=sys.stderr)
        return None


def find_exact_duplicates(files):
    grouped_by_hash = defaultdict(list)
    for path in files:
        grouped_by_hash[file_sha256(path)].append(path)
    return [group for group in grouped_by_hash.values() if len(group) > 1]


def find_similar_videos(files, sample_count=8, hash_size=16, threshold=25.0):
    signatures = {}
    for path in files:
        try:
            signatures[path] = sample_frame_hashes(path, sample_count=sample_count, hash_size=hash_size)
        except Exception as exc:
            print(f"Warning: could not process {path}: {exc}", file=sys.stderr)
            signatures[path] = []

    similar_pairs = []
    for index, path_a in enumerate(files):
        for path_b in files[index + 1 :]:
            distance = signature_distance(signatures[path_a], signatures[path_b])
            if distance <= threshold:
                similar_pairs.append((path_a, path_b, round(distance, 2)))
    return similar_pairs


def find_content_duplicate_groups(files, sample_count=8, hash_size=16, video_threshold=6.0, audio_threshold=0.12):
    video_signatures = {}
    audio_signatures = {}

    for path in files:
        try:
            video_signatures[path] = sample_frame_hashes(path, sample_count=sample_count, hash_size=hash_size)
        except Exception as exc:
            print(f"Warning: could not build video signature for {path}: {exc}", file=sys.stderr)
            video_signatures[path] = []

        try:
            audio_signatures[path] = sample_audio_signature(path)
        except Exception as exc:
            print(f"Warning: could not build audio signature for {path}: {exc}", file=sys.stderr)
            audio_signatures[path] = None

    parent = {path: path for path in files}

    def find(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(path_a, path_b):
        root_a = find(path_a)
        root_b = find(path_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for index, path_a in enumerate(files):
        for path_b in files[index + 1 :]:
            video_distance = signature_distance(video_signatures[path_a], video_signatures[path_b])
            if video_distance > video_threshold:
                continue

            audio_distance = numeric_signature_distance(audio_signatures[path_a], audio_signatures[path_b])
            if audio_distance <= audio_threshold:
                union(path_a, path_b)

    grouped_paths = defaultdict(list)
    for path in files:
        grouped_paths[find(path)].append(path)

    return [group for group in grouped_paths.values() if len(group) > 1]


def export_unique_videos(files, duplicate_groups, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    representative_by_path = {}
    for group in duplicate_groups:
        representative = group[0]
        for path in group:
            representative_by_path[path] = representative

    exported_files = []
    skipped_duplicates = []
    seen_representatives = set()

    for index, path in enumerate(files, start=1):
        representative = representative_by_path.get(path, path)
        if representative in seen_representatives:
            skipped_duplicates.append((path, representative))
            continue

        seen_representatives.add(representative)
        extension = os.path.splitext(representative)[1]
        output_name = f"{index:03d}_{os.path.basename(representative)}"
        output_path = os.path.join(output_folder, output_name)
        shutil.copy2(representative, output_path)
        exported_files.append((representative, output_path))

    return exported_files, skipped_duplicates


def print_exact_duplicates(duplicate_groups):
    if not duplicate_groups:
        print("No exact duplicate videos found.")
        return

    print("Exact duplicate video groups:")
    for index, group in enumerate(duplicate_groups, start=1):
        print(f"\nGroup {index}:")
        for path in group:
            print(f"  {path}")


def print_similar_videos(similar_pairs):
    if not similar_pairs:
        print("No similar videos found.")
        return

    print("Similar video pairs:")
    for path_a, path_b, distance in similar_pairs:
        print(f"  [{distance}] {path_a}")
        print(f"       {path_b}")


def print_content_duplicate_groups(duplicate_groups):
    if not duplicate_groups:
        print("No content-duplicate videos found with matching video and audio.")
        return

    print("Content duplicate groups (matching video and audio):")
    for index, group in enumerate(duplicate_groups, start=1):
        print(f"\nGroup {index}:")
        for path in group:
            print(f"  {path}")


def write_report(output_path, duplicate_groups, similar_pairs, content_duplicate_groups=None, exported_files=None, skipped_duplicates=None):
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("Exact duplicate video groups:\n")
        if duplicate_groups:
            for index, group in enumerate(duplicate_groups, start=1):
                output_file.write(f"\nGroup {index}:\n")
                for path in group:
                    output_file.write(f"{path}\n")
        else:
            output_file.write("No exact duplicate videos found.\n")

        output_file.write("\nSimilar video pairs:\n")
        if similar_pairs:
            for path_a, path_b, distance in similar_pairs:
                output_file.write(f"[{distance}] {path_a}\n")
                output_file.write(f"{path_b}\n\n")
        else:
            output_file.write("No similar videos found.\n")

        output_file.write("\nContent duplicate groups (matching video and audio):\n")
        if content_duplicate_groups:
            for index, group in enumerate(content_duplicate_groups, start=1):
                output_file.write(f"\nGroup {index}:\n")
                for path in group:
                    output_file.write(f"{path}\n")
        else:
            output_file.write("No content-duplicate videos found with matching video and audio.\n")

        output_file.write("\nExported unique files:\n")
        if exported_files:
            for source_path, exported_path in exported_files:
                output_file.write(f"{source_path}\n")
                output_file.write(f"  {exported_path}\n")
        else:
            output_file.write("No unique export was created.\n")

        output_file.write("\nSkipped duplicate copies:\n")
        if skipped_duplicates:
            for path, representative in skipped_duplicates:
                output_file.write(f"{path}\n")
                output_file.write(f"  kept: {representative}\n")
        else:
            output_file.write("No duplicate copies were skipped.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Find duplicated and similar videos in a folder.")
    parser.add_argument("folder", help="Folder to scan for videos")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of frames to sample per video")
    parser.add_argument("--hash-size", type=int, default=16, help="Perceptual hash size for frames")
    parser.add_argument("--threshold", type=float, default=25.0, help="Similarity threshold for video pairs")
    parser.add_argument("--extensions", nargs="*", help="Additional video file extensions to include")
    parser.add_argument("--output", "-o", help="Write results to a text file")
    parser.add_argument("--content-video-threshold", type=float, default=6.0, help="Video signature threshold for content duplicate grouping")
    parser.add_argument("--content-audio-threshold", type=float, default=0.12, help="Audio signature threshold for content duplicate grouping")
    parser.add_argument("--export-unique-folder", help="Copy one representative video per content-duplicate group into this folder")
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

    files = collect_video_files(args.folder)
    if not files:
        print("No video files found in the given folder.")
        return

    print(f"Scanning {len(files)} video file(s) in: {args.folder}")

    print("\nChecking exact duplicates...")
    duplicate_groups = find_exact_duplicates(files)
    print_exact_duplicates(duplicate_groups)

    print("\nChecking visually similar videos...")
    similar_pairs = find_similar_videos(
        files,
        sample_count=args.sample_count,
        hash_size=args.hash_size,
        threshold=args.threshold,
    )
    print_similar_videos(similar_pairs)

    print("\nChecking content duplicates with matching video and audio...")
    content_duplicate_groups = find_content_duplicate_groups(
        files,
        sample_count=args.sample_count,
        hash_size=args.hash_size,
        video_threshold=args.content_video_threshold,
        audio_threshold=args.content_audio_threshold,
    )
    print_content_duplicate_groups(content_duplicate_groups)

    exported_files = None
    skipped_duplicates = None
    if args.export_unique_folder:
        exported_files, skipped_duplicates = export_unique_videos(files, content_duplicate_groups, args.export_unique_folder)
        print(f"\nUnique content videos written to: {args.export_unique_folder}")
        for source_path, exported_path in exported_files:
            print(f"  {source_path}")
            print(f"       {exported_path}")
        if skipped_duplicates:
            print("Skipped duplicate copies:")
            for path, representative in skipped_duplicates:
                print(f"  {path}")
                print(f"       kept: {representative}")

    if args.output:
        try:
            write_report(
                args.output,
                duplicate_groups,
                similar_pairs,
                content_duplicate_groups=content_duplicate_groups,
                exported_files=exported_files,
                skipped_duplicates=skipped_duplicates,
            )
            print(f"\nResults written to: {args.output}")
        except Exception as exc:
            print(f"Failed to write output file: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()