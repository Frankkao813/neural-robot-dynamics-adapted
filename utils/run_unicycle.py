"""Standalone demo: python utils/run_unicycle.py --speed 1 --turn-rate 0.5.

Use --headless for a numerical rollout without opening a renderer.
Use --export-video-path outputs/unicycle.gif with an OpenGL display (such as
an Xvfb display in a batch job) to save an animation using FFmpeg.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.warp_sim_envs.env_unicycle import RenderMode, UnicycleEnvironment


def encode_gif(frames_dir, output_path, fps, ffmpeg):
    """Encode numbered PNG frames with a shared palette; propagate failures."""
    frames_dir = Path(frames_dir)
    if not (frames_dir / "000000.png").is_file():
        raise RuntimeError("No video frames were captured.")
    sequence = str(frames_dir / "%06d.png")
    palette = str(frames_dir / "palette.png")
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-framerate", str(fps), "-i", sequence,
         "-vf", "palettegen", "-frames:v", "1", palette], check=True,
    )
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
         "-framerate", str(fps), "-i", sequence, "-i", palette,
         "-lavfi", "paletteuse", "-loop", "0", str(output_path)], check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.0, help="Forward speed in m/s.")
    parser.add_argument("--turn-rate", type=float, default=0.5, help="Turning rate in rad/s toward +Z.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--export-video-path", type=Path, help="Output GIF path; requires FFmpeg and rendering.")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    ffmpeg = None
    if args.export_video_path is not None:
        if args.headless:
            parser.error("--headless disables rendering; omit it when exporting video under Xvfb")
        if args.export_video_path.suffix.lower() != ".gif":
            parser.error("--export-video-path must end in .gif")
        if args.export_video_path.exists():
            parser.error(f"Output already exists: {args.export_video_path}")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            parser.error("FFmpeg was not found on PATH; load the FFmpeg module before exporting")
        args.export_video_path.parent.mkdir(parents=True, exist_ok=True)
    env = UnicycleEnvironment(
        num_envs=args.num_envs, device=args.device,
        render_mode=RenderMode.NONE if args.headless else RenderMode.OPENGL,
        setup_renderer=False,
    )
    frame_folder = None
    try:
        # An elevated camera keeps the default 2 m radius circle in view.
        # Warp camera positions are in scaled renderer coordinates.
        env.opengl_render_settings = dict(
            scaling=3.0, screen_width=1280, screen_height=720,
            camera_pos=(0.0, 12.0, 22.0), camera_front=(0.0, -0.6, -1.0),
            device=env.device,
        )
        env.setup_renderer()
        pixels = None
        if args.export_video_path is not None:
            import cv2

            frame_folder = tempfile.TemporaryDirectory(
                prefix="unicycle_frames_", dir=str(args.export_video_path.parent),
            )
            pixels = wp.empty(
                (env.renderer.screen_height, env.renderer.screen_width, 3),
                dtype=wp.uint8, device=env.device,
            )
        frame_count = 0
        env.set_commands([[args.speed, args.turn_rate]] * args.num_envs)
        for _ in range(args.steps):
            if env.renderer is not None and not env.renderer.is_running():
                break
            env.update()
            env.render()
            if pixels is not None:
                if not env.renderer.get_pixels(pixels, split_up_tiles=False, mode="rgb", use_uint8=True):
                    raise RuntimeError(f"Failed to capture video frame {frame_count}.")
                frame_path = Path(frame_folder.name) / f"{frame_count:06d}.png"
                if not cv2.imwrite(str(frame_path), pixels.numpy()[:, :, ::-1]):
                    raise RuntimeError(f"Failed to write video frame: {frame_path}")
                frame_count += 1
        print("Final local poses (x, z, theta):")
        print(env.get_pose())
        if frame_folder is not None:
            encode_gif(frame_folder.name, args.export_video_path, 1.0 / env.frame_dt, ffmpeg)
            print(f"Saved {frame_count} frames to {args.export_video_path.resolve()}")
    finally:
        env.close()
        if frame_folder is not None:
            frame_folder.cleanup()


if __name__ == "__main__":
    main()
