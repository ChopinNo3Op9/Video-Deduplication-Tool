# Video Deduplication Tool

A small Python utility to find duplicate and visually similar video files in a directory.

## Features

- Detects exact duplicate videos by computing SHA-256 file hashes.
- Detects visually similar videos by sampling frames and comparing perceptual image hashes.
- Supports common video formats such as `.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.webm`, `.mpg`, and `.mpeg`.
- Optional output report generation.
- Extendable with additional file extensions via command line.

## Requirements

- Python 3.7+
- `opencv-python`
- `numpy`
- `pillow`
- `imagehash`

Install dependencies with:

```bash
pip install opencv-python numpy pillow imagehash
```

## Usage

Run the scanner on a folder containing videos:

```bash
python find_video_duplicates.py /path/to/folder
```

Optional arguments:

- `--sample-count`: Number of frames sampled per video (default: 8)
- `--hash-size`: Perceptual hash size for frames (default: 16)
- `--threshold`: Similarity threshold for video pairs (default: 25.0)
- `--extensions`: Additional video extensions to include
- `--output`, `-o`: Write results to a text file

Example:

```bash
python find_video_duplicates.py C:\Videos --sample-count 10 --threshold 20 --output duplicate_report.txt
```

## Notes

- Exact duplicates are detected only when files are identical at the binary level.
- Similarity detection compares sampled frames and may report videos with visually close content.
- Processing large video collections can take time depending on file count and sample settings.

## License

This project is available under the terms of the included `LICENSE` file.
