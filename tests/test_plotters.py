import os
import tempfile
import shutil
import pytest
import imageio
import plotly.graph_objects as go
from jaxidem.plotters import save_st_gif  # Replace with actual import


def generate_sample_frames(n=5):
    frames = []
    for i in range(n):
        trace = go.Scatter(x=[0, 1, 2], y=[i, i + 1, i + 2], mode="lines+markers")
        frame = go.Figure(data=[trace])
        frames.append(frame)
    return frames


def dummy_theme(fig, fontsize=20):
    fig.update_layout(template="plotly_white", font=dict(size=fontsize))


def test_save_st_gif_creates_valid_gif_and_cleans_up():
    frames = generate_sample_frames()
    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = os.path.join(tmpdir, "test.gif")

        save_st_gif(frames, gif_path, apply_theme=dummy_theme)

        # Check that GIF was created
        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0

        # Check that temp folder was removed
        assert not os.path.exists("frames")

        # Optionally: verify GIF is readable
        gif = imageio.mimread(gif_path)
        assert len(gif) == len(frames)
