# Video Deduplication Tool

A Python utility to scan a folder of videos, report duplicates and similar clips, and optionally combine or trim videos.

## Features

- Detects exact duplicate videos by computing SHA-256 file hashes.
- Detects visually similar videos by sampling frames and comparing perceptual image hashes.
- Includes a dedicated deduplication-only script for lightweight scanning.
- Combines all unique videos in the folder into one output file.
- Orders the combined video by audio pace from slow to fast using the original soundtrack of each clip.
- Includes a dedicated stitcher that repeats faster clips more times in the final sequence.
- Trims only exactly repeated sampled segments from later clips even when the reused segment appears in the middle of a video.
- Preserves audio in the combined output.
- Can export a new folder of per-video outputs with leading overlap trimmed.
- Supports common video formats such as `.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.webm`, `.mpg`, and `.mpeg`.
- Optional output report generation.
- Extendable with additional file extensions via command line.

## Requirements

- Python 3.7+
- `opencv-python`
- `numpy`
- `pillow`
- `imagehash`
- `moviepy`

Install dependencies with:

```bash
pip install opencv-python numpy pillow imagehash moviepy
```

## Usage

Deduplication-only scanner:

```bash
python video_deduplication.py /path/to/folder
```

Deduplication-only optional arguments:

- `--sample-count`: Number of frames sampled per video (default: 8)
- `--hash-size`: Perceptual hash size for frames (default: 16)
- `--threshold`: Similarity threshold for video-only similar pairs (default: 25.0)
- `--content-video-threshold`: Video threshold for content duplicate grouping (default: 6.0)
- `--content-audio-threshold`: Audio threshold for content duplicate grouping (default: 0.12)
- `--export-unique-folder`: Copy one representative video per matching video+audio group into this folder
- `--output`, `-o`: Write results to a text file

Full workflow script:

```bash
python find_video_duplicates.py /path/to/folder
```

Pace-based stitcher:

```bash
python stitch_videos_by_pace.py /path/to/folder --output stitched.mp4
```

Optional arguments:

- `--sample-count`: Number of frames sampled per video (default: 8)
- `--hash-size`: Perceptual hash size for frames (default: 16)
- `--threshold`: Similarity threshold for video pairs (default: 25.0)
- `--extensions`: Additional video extensions to include
- `--output`, `-o`: Write results to a text file
- `--create-edit`: Write one combined video to this path
- `--export-trimmed-folder`: Write trimmed per-video outputs to this folder
- `--max-edit-clips`: Optional maximum number of unique videos to combine (default: 0 = all)
- `--clip-seconds`: Optional maximum seconds to use from each input video (default: 0 = full video)
- `--overlap-seconds`: Minimum repeated segment length to detect and trim (default: 1.5)
- `--scan-step-seconds`: Sliding scan step for arbitrary timestamp matching (default: overlap-seconds / 4)
- `--edit-height`: Output height of the combined video (default: 720)
- `--edit-fps`: Output frame rate of the combined video (default: 30)

Pace-based stitcher optional arguments:

- `--output`: Output stitched video path
- `--report`: Optional text report path
- `--clip-seconds`: Optional maximum seconds to use from each clip (default: 0 = full video)
- `--height`: Output height (default: 720)
- `--fps`: Output frame rate (default: 30)
- `--extensions`: Additional video extensions to include

Example:

```bash
python video_deduplication.py C:\Videos --sample-count 10 --threshold 20 --output duplicate_report.txt
```

Export one kept copy per matching video+audio duplicate group:

```bash
python video_deduplication.py C:\Videos --export-unique-folder C:\Videos_unique --output duplicate_report.txt
```

Combine the whole folder into one video ordered by audio pace:

```bash
python find_video_duplicates.py C:\Videos --create-edit combined.mp4 --overlap-seconds 1.5 --output duplicate_report.txt
```

Create a stitched edit that starts slower and repeats the faster clips more often:

```bash
python stitch_videos_by_pace.py C:\Videos --output stitched.mp4 --report stitch_report.txt
```

Export a new folder of trimmed videos with up to 10 seconds of leading overlap removed:

```bash
python find_video_duplicates.py C:\Videos --export-trimmed-folder C:\Videos_trimmed --overlap-seconds 10 --output trim_report.txt
```

## Combine Behavior

- Exact duplicates are collapsed before combination so identical files are not repeated.
- The combine order is based on an audio pace score derived from the clip's original soundtrack, from slow to fast.
- The script compares the tail of each clip to the head of the next one and trims detected visual overlap from the following clip.
- The output is written as an `.mp4` file with original audio preserved.
- Unreadable or corrupted clips are skipped and listed in the report instead of aborting the whole combine.

## Pace Stitch Behavior

- Exact duplicate files are collapsed before stitching so the same source file is not included twice by accident.
- Videos are sorted by audio pace from slow to fast.
- Repeat counts are assigned from the relative pace score: slow clips usually play once, faster clips step up to 2x, 3x, or 4x.
- The stitcher preserves original audio for each repeated section.

## Trimmed Folder Behavior

- Videos are processed in folder order.
- Each video is compared against earlier videos using sliding windows over the full clip duration.
- Reused content can be detected at arbitrary timestamps, including repeated sections in the middle of a clip.
- A repeated window is only accepted when all sampled frames match exactly by raw-frame digest and the full candidate period passes a frame-by-frame verification.
- Matched ranges are removed from the later video and the remaining sections are stitched back together.
- Untouched clips keep their original container extension when copied as-is.
- Trimmed clips are written as `.mp4` files with audio preserved.

## Notes

- Exact duplicates are detected only when files are identical at the binary level.
- Similarity detection compares sampled frames and may report videos with visually close content.
- Content-duplicate grouping in `video_deduplication.py` requires both sampled video and sampled audio to match within threshold.
- Exact-window trimming still depends on the sampled frames, window size, and scan step. If two clips contain the same scene but were re-encoded, resized, or filtered differently, they will usually not count as exact matches.
- Processing large video collections can take time depending on file count and sample settings.

## License

This project is available under the terms of the included `LICENSE` file.
