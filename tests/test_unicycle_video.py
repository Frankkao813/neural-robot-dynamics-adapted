"""Exercise the batch video encoder without requiring an OpenGL display."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from utils.run_unicycle import encode_gif, main


class UnicycleVideoTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools required")
    def test_encode_gif_frames_and_timing(self):
        with tempfile.TemporaryDirectory(prefix="unicycle_video_test_") as folder:
            root = Path(folder)
            frames = root / "frames with spaces"
            frames.mkdir()
            for i in range(4):
                pixels = np.zeros((32, 48, 3), dtype=np.uint8)
                pixels[:, i * 8:i * 8 + 8, 2] = 255
                self.assertTrue(cv2.imwrite(str(frames / f"{i:06d}.png"), pixels))
            output = root / "video with spaces.gif"
            encode_gif(frames, output, 10, shutil.which("ffmpeg"))
            result = subprocess.run(
                [shutil.which("ffprobe"), "-v", "error", "-count_frames",
                 "-show_entries", "stream=width,height,nb_read_frames,duration",
                 "-of", "json", str(output)], check=True, capture_output=True, text=True,
            )
            stream = json.loads(result.stdout)["streams"][0]
            self.assertEqual((stream["width"], stream["height"]), (48, 32))
            self.assertEqual(int(stream["nb_read_frames"]), 4)
            self.assertAlmostEqual(float(stream["duration"]), 0.4, places=2)
            original = output.read_bytes()
            with self.assertRaises(subprocess.CalledProcessError):
                encode_gif(frames, output, 10, shutil.which("ffmpeg"))
            self.assertEqual(output.read_bytes(), original)

    def test_empty_capture_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "No video frames"):
                encode_gif(folder, Path(folder) / "empty.gif", 60, "ffmpeg")

    def test_headless_export_is_rejected_before_simulation(self):
        with patch("sys.argv", ["run_unicycle.py", "--headless", "--export-video-path", "video.gif"]):
            with patch("utils.run_unicycle.UnicycleEnvironment") as environment:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
