#!/usr/bin/env python3
"""
Trim reused content from videos in a folder.

The script scans each full video using sliding time windows. If a later video contains
content that matches any segment from an earlier video, the matched ranges are removed
from the later video and the rebuilt result is written to a new folder.

Usage:
    python trim_video_overlaps.py /path/to/folder --output /path/to/output_folder --overlap-seconds 10

Requirements:
    pip install opencv-python pillow imagehash numpy moviepy
"""

import argparse
import hashlib
import os
import shutil

import cv2
from PIL import Image
import imagehash
from moviepy.editor import VideoFileClip, concatenate_videoclips

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm", ".mpg"}

def is_video_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS

def collect_video_files(folder):
    video_files = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            if is_video_file(path):
                video_files.append(path)
    return video_files

def sample_frame_hashes(path, sample_count=8, hash_size=16, start_time=0.0, end_time=None):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    duration = frame_count / fps if fps > 0 else 0
    if frame_count <= 0 or duration <= 0:
        capture.release()
        return []
    analysis_start = max(0.0, min(start_time, duration))
    analysis_end = duration if end_time is None else max(analysis_start, min(end_time, duration))
    if analysis_end <= analysis_start:
        capture.release()
        return []
    positions = [
        int(fps * (analysis_start + (analysis_end - analysis_start) * index / (sample_count + 1)))
        for index in range(1, sample_count + 1)
    ]
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

def sample_frame_digests(path, sample_count=8, start_time=0.0, end_time=None):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    duration = frame_count / fps if fps > 0 else 0
    if frame_count <= 0 or duration <= 0:
        capture.release()
        return []
    analysis_start = max(0.0, min(start_time, duration))
    analysis_end = duration if end_time is None else max(analysis_start, min(end_time, duration))
    if analysis_end <= analysis_start:
        capture.release()
        return []
    positions = [
        int(fps * (analysis_start + (analysis_end - analysis_start) * index / (sample_count + 1)))
        for index in range(1, sample_count + 1)
    ]
    digests = []
    for frame_idx in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        if not success or frame is None:
            continue
        digests.append(hashlib.sha256(frame.tobytes()).hexdigest())
    capture.release()
    return digests

def signature_distance(signature_a, signature_b):
    if not signature_a or not signature_b:
        return float("inf")
    length = min(len(signature_a), len(signature_b))
    distances = [(signature_a[index] - signature_b[index]) for index in range(length)]
    return sum(distances) / len(distances)

def exact_digest_match(digests_a, digests_b):
    if not digests_a or not digests_b:
        return False
    if len(digests_a) != len(digests_b):
        return False
    return all(digest_a == digest_b for digest_a, digest_b in zip(digests_a, digests_b))

def exact_frame_sequence_match(video_a, start_a, end_a, video_b, start_b, end_b):
    capture_a = cv2.VideoCapture(video_a)
    capture_b = cv2.VideoCapture(video_b)
    if not capture_a.isOpened() or not capture_b.isOpened():
        capture_a.release()
        capture_b.release()
        return False

    fps_a = capture_a.get(cv2.CAP_PROP_FPS) or 0.0
    fps_b = capture_b.get(cv2.CAP_PROP_FPS) or 0.0
    if fps_a <= 0 or fps_b <= 0:
        capture_a.release()
        capture_b.release()
        return False

    start_frame_a = int(round(start_a * fps_a))
    end_frame_a = int(round(end_a * fps_a))
    start_frame_b = int(round(start_b * fps_b))
    end_frame_b = int(round(end_b * fps_b))
    frame_total_a = max(0, end_frame_a - start_frame_a)
    frame_total_b = max(0, end_frame_b - start_frame_b)

    if frame_total_a == 0 or frame_total_b == 0 or frame_total_a != frame_total_b:
        capture_a.release()
        capture_b.release()
        return False

    capture_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame_a)
    capture_b.set(cv2.CAP_PROP_POS_FRAMES, start_frame_b)

    for _ in range(frame_total_a):
        success_a, frame_a = capture_a.read()
        success_b, frame_b = capture_b.read()
        if not success_a or not success_b or frame_a is None or frame_b is None:
            capture_a.release()
            capture_b.release()
            return False
        if frame_a.shape != frame_b.shape or not (frame_a == frame_b).all():
            capture_a.release()
            capture_b.release()
            return False

    capture_a.release()
    capture_b.release()
    return True

def merge_ranges(ranges, gap_tolerance=0.05):
    """Merge overlapping or adjacent time ranges with a small gap tolerance (seconds)."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda item: (item[0], item[1]))
    merged = [list(sorted_ranges[0])]
    for start_time, end_time in sorted_ranges[1:]:
        last_range = merged[-1]
        if start_time <= last_range[1] + gap_tolerance:
            last_range[1] = max(last_range[1], end_time)
        else:
            merged.append([start_time, end_time])
    return [(start_time, end_time) for start_time, end_time in merged]

def detect_matching_ranges(video_a, signatures_a, video_b, signatures_b, threshold=6.0):
    matched_ranges = []
    matches = []
    for window_b in signatures_b:
        best_distance = float("inf")
        best_window_a = None
        for window_a in signatures_a:
            distance = signature_distance(window_a["signature"], window_b["signature"])
            if distance < best_distance:
                best_distance = distance
                best_window_a = window_a
        if best_window_a is not None and best_distance <= threshold:
            matched_ranges.append((window_b["start"], window_b["end"]))
            matches.append(
                {
                    "source_start": best_window_a["start"],
                    "source_end": best_window_a["end"],
                    "target_start": window_b["start"],
                    "target_end": window_b["end"],
                    "distance": best_distance,
                }
            )
    return merge_ranges(matched_ranges), matches

def trim_video_ranges(input_path, output_path, remove_ranges):
    with VideoFileClip(input_path) as clip:
        duration = clip.duration or 0
        merged_ranges = []
        for start_time, end_time in merge_ranges(remove_ranges):
            bounded_start = max(0.0, min(start_time, duration))
            bounded_end = max(bounded_start, min(end_time, duration))
            if bounded_end > bounded_start:
                merged_ranges.append((bounded_start, bounded_end))

        if not merged_ranges:
            clip.write_videofile(output_path, audio=True, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            return True

        if merged_ranges[0][0] <= 0.0 and merged_ranges[-1][1] >= duration and len(merged_ranges) == 1:
            print(f"Warning: {input_path} would be fully removed, skipping.")
            return False

        kept_segments = []
        current_start = 0.0
        for start_time, end_time in merged_ranges:
            if start_time > current_start:
                kept_segments.append(clip.subclip(current_start, start_time))
            current_start = max(current_start, end_time)
        if current_start < duration:
            kept_segments.append(clip.subclip(current_start, duration))

        kept_segments = [segment for segment in kept_segments if (segment.duration or 0) > 0.05]
        if not kept_segments:
            print(f"Warning: {input_path} has no content left after trimming, skipping.")
            return False
        else:
            output_clip = concatenate_videoclips(kept_segments)
            output_clip.write_videofile(output_path, audio=True, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            output_clip.close()
            for segment in kept_segments:
                segment.close()
    return True

def build_window_signatures(path, window_seconds, hash_size, step_seconds, sample_count=8):
    try:
        with VideoFileClip(path) as clip:
            duration = clip.duration or 0
    except Exception:
        return [], 0.0
    if duration < window_seconds:
        return [], duration
    if step_seconds <= 0:
        raise ValueError("step_seconds must be greater than zero")

    signatures = []
    start_time = 0.0
    last_start = max(0.0, duration - window_seconds)
    while start_time <= last_start + 1e-6:
        end_time = min(duration, start_time + window_seconds)
        signature = sample_frame_hashes(
            path,
            sample_count=sample_count,
            hash_size=hash_size,
            start_time=start_time,
            end_time=end_time,
        )
        if signature:
            signatures.append({"start": start_time, "end": end_time, "signature": signature})
        start_time += step_seconds
    if signatures and signatures[-1]["end"] < duration:
        start_time = last_start
        signature = sample_frame_hashes(
            path,
            sample_count=sample_count,
            hash_size=hash_size,
            start_time=start_time,
            end_time=duration,
        )
        if signature:
            signatures.append({"start": start_time, "end": duration, "signature": signature})
    return signatures, duration

def main():
    parser = argparse.ArgumentParser(description="Trim reused video segments found at arbitrary timestamps across a folder.")
    parser.add_argument("folder", help="Folder to scan for videos")
    parser.add_argument("--output", required=True, help="Output folder for trimmed videos")
    parser.add_argument("--overlap-seconds", type=float, default=10.0, help="Minimum repeated segment length to detect and trim")
    parser.add_argument("--hash-size", type=int, default=16, help="Perceptual hash size for frames")
    parser.add_argument("--threshold", type=float, default=6.0, help="Signature distance threshold for overlap detection")
    parser.add_argument("--scan-step-seconds", type=float, default=0.0, help="Sliding window step size in seconds; defaults to overlap-seconds / 4")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = collect_video_files(args.folder)
    print(f"Scanning {len(files)} video file(s) in: {args.folder}")
    trimmed = set()
    overlap_report = []
    trimmed_report = []
    copied_report = []
    trim_ranges_by_video = {}
    step_seconds = args.scan_step_seconds if args.scan_step_seconds > 0 else max(0.5, args.overlap_seconds / 4.0)
    signature_cache = {}

    for path in files:
        signatures, duration = build_window_signatures(
            path,
            window_seconds=args.overlap_seconds,
            hash_size=args.hash_size,
            step_seconds=step_seconds,
        )
        signature_cache[path] = {"signatures": signatures, "duration": duration}

    for target_index, video_b in enumerate(files):
        target_signatures = signature_cache[video_b]["signatures"]
        if not target_signatures:
            continue
        for source_index in range(target_index):
            video_a = files[source_index]
            source_signatures = signature_cache[video_a]["signatures"]
            if not source_signatures:
                continue
            matched_ranges, matches = detect_matching_ranges(video_a, source_signatures, video_b, target_signatures, threshold=args.threshold)
            if not matched_ranges:
                continue
            trim_ranges_by_video.setdefault(video_b, []).extend(matched_ranges)
            for match in matches:
                msg = (
                    f"Match detected: {os.path.basename(video_a)} "
                    f"[{match['source_start']:.2f}-{match['source_end']:.2f}s] -> "
                    f"{os.path.basename(video_b)} "
                    f"[{match['target_start']:.2f}-{match['target_end']:.2f}s] "
                    f"(distance={match['distance']:.2f})"
                )
                print(msg)
                overlap_report.append(msg)

    for path in files:
        output_path = os.path.join(args.output, os.path.basename(path))
        trim_ranges = merge_ranges(trim_ranges_by_video.get(path, []), gap_tolerance=step_seconds / 2.0)
        if trim_ranges:
            if trim_video_ranges(path, output_path, trim_ranges):
                trimmed.add(path)
                trimmed_report.append(
                    f"{os.path.basename(path)} | removed ranges: "
                    + ", ".join(f"{start_time:.2f}-{end_time:.2f}s" for start_time, end_time in trim_ranges)
                )
            else:
                shutil.copy2(path, output_path)
                copied_report.append(os.path.basename(path))
        else:
            shutil.copy2(path, output_path)
            copied_report.append(os.path.basename(path))
    # Copy untrimmed videos
    print(f"Trimmed videos written to: {args.output}")

    # Write report
    report_path = os.path.join(args.output, "trim_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Trim Video Overlaps Report\n")
        f.write(f"Input folder: {args.folder}\n")
        f.write(f"Output folder: {args.output}\n\n")
        f.write(f"Window matches detected ({len(overlap_report)}):\n")
        for line in overlap_report:
            f.write(line + "\n")
        f.write("\nTrimmed Videos ({}):\n".format(len(trimmed_report)))
        for name in trimmed_report:
            f.write(name + "\n")
        f.write("\nCopied Videos ({}):\n".format(len(copied_report)))
        for name in copied_report:
            f.write(name + "\n")
    print(f"Report written to: {report_path}")

if __name__ == "__main__":
    main()
