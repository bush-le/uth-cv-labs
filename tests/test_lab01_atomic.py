"""Tests for Lab 01 atomic modules."""

from pathlib import Path
import numpy as np
import pytest

from src.labs.lab01_basics import color, geometry, drawing, io, visuals


def test_color_conversions():
    dummy = np.zeros((20, 20, 3), dtype=np.uint8)
    dummy[:, :, 0] = 255  # Pure Blue in BGR
    gray = color.bgr_to_grayscale(dummy)
    assert gray.shape == (20, 20)
    assert gray.dtype == np.uint8

    hsv = color.bgr_to_hsv(dummy)
    assert hsv.shape == (20, 20, 3)

    lab = color.bgr_to_lab(dummy)
    assert lab.shape == (20, 20, 3)

    rgb = color.bgr_to_rgb(dummy)
    assert rgb[0, 0, 2] == 255  # Blue channel in RGB


def test_geometry_crop_and_resize():
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped = geometry.crop_roi(dummy, 10, 10, 40, 40)
    assert cropped.shape == (30, 30, 3)

    resized_dim = geometry.resize_by_dimensions(dummy, 50, 60)
    assert resized_dim.shape == (60, 50, 3)

    resized_scale = geometry.resize_by_scale(dummy, 0.5)
    assert resized_scale.shape == (50, 50, 3)


def test_drawing_primitives():
    canvas = np.zeros((50, 50, 3), dtype=np.uint8)
    rect = drawing.draw_rectangle(canvas, (5, 5), (20, 20), (0, 255, 0), 2)
    assert rect.shape == (50, 50, 3)

    circle = drawing.draw_circle(canvas, (25, 25), 10, (0, 0, 255), -1)
    assert circle.shape == (50, 50, 3)

    line = drawing.draw_line(canvas, (0, 0), (49, 49), (255, 255, 255), 1)
    assert line.shape == (50, 50, 3)

    text = drawing.draw_text(canvas, "Test", (10, 10), (255, 255, 255))
    assert text.shape == (50, 50, 3)


def test_visuals_creation():
    dummy = np.zeros((30, 30, 3), dtype=np.uint8)
    gray = np.zeros((30, 30), dtype=np.uint8)
    fig1 = visuals.plot_color_spaces(dummy, gray, dummy, dummy)
    assert fig1 is not None

    fig2 = visuals.plot_roi_and_resize(dummy, dummy, dummy, dummy)
    assert fig2 is not None
