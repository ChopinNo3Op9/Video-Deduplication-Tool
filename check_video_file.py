from moviepy.editor import VideoFileClip
import sys

try:
    path = sys.argv[1]
    clip = VideoFileClip(path)
    print(f"duration: {clip.duration}")
    print(f"fps: {clip.fps}")
    print(f"size: {clip.size}")
    print(f"audio: {clip.audio is not None}")
    clip.close()
except Exception as e:
    print(f"ERROR: {e}")
