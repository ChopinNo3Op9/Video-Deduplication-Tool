#!/usr/bin/env python3
"""Find duplicate and similar videos in a folder.

Exact duplicates are detected by file hash. Similar videos are detected by sampling frames
and comparing perceptual image hashes.

Usage:
    python find_video_duplicates.py /path/to/folder

Requirements:
    pip install opencv-python numpy pillow imagehash
"""

import argparse
import hashlib
import os
import sys
from collections import defaultdict

try:
    import cv2
    from PIL import Image
    import imagehash
except ImportError as exc:
    print("Missing dependency: {}".format(exc), file=sys.stderr)
    print("Install required packages with: pip install opencv-python numpy pillow imagehash", file=sys.stderr)
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


def find_exact_duplicates(files):
    hash_groups = defaultdict(list)
    for path in files:
        file_hash = file_sha256(path)
        hash_groups[file_hash].append(path)

    return [group for group in hash_groups.values() if len(group) > 1]


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
    if not similar_pairs:
        print("No similar videos found.")
        return

    print("Similar video pairs:")
    for path_a, path_b, distance in similar_pairs:
        print(f"  [{distance}] {path_a}")
        print(f"       {path_b}")


def parse_args():
    parser = argparse.ArgumentParser(description="Find duplicated and similar videos in a folder.")
    parser.add_argument("folder", help="Folder to scan for videos")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of frames to sample per video")
    parser.add_argument("--hash-size", type=int, default=16, help="Perceptual hash size for frames")
    parser.add_argument("--threshold", type=float, default=25.0, help="Similarity threshold for video pairs")
    parser.add_argument("--extensions", nargs="*", help="Additional video file extensions to include")
    parser.add_argument("--output", "-o", help="Write results to a text file")
    return parser.parse_args()


def write_report(output_path, duplicate_groups, similar_pairs):
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
        if similar_pairs:
            for path_a, path_b, distance in similar_pairs:
                out.write(f"[{distance}] {path_a}\n")
                out.write(f"{path_b}\n\n")
        else:
            out.write("No similar videos found.\n")


def main():
    args = parse_args()

    if args.extensions:
        for ext in args.extensions:
            normalized = ext if ext.startswith(".") else f".{ext}"
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

    if args.output:
        try:
            write_report(args.output, duplicate_groups, similar_pairs)
            print(f"\nResults written to: {args.output}")
        except Exception as exc:
            print(f"Failed to write output file: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
