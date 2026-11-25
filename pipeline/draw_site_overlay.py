#!/usr/bin/env python3
"""
Draw a site's processing settings on top of one camera image, to check they still fit.

Shows the PCS lines, background (ambient) ROI, optical flow ROI and the spectrometer field
of view from a PyCam config. Light dilution lines are only drawn when use_light_dilution is
on, and the source point only when flow_nadeau is on.

Example:
    python pipeline/draw_site_overlay.py --config sites/reventador_template.yml \
        --image 2026-11-24T123000_fltrA_1.0_2999593ss_Plume.png --out overlay.png
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import yaml
from PIL import Image


def find_file(path, config_dir):
    """Config paths can be absolute, relative to the current folder or relative to the config."""
    if os.path.isabs(path) or os.path.exists(path):
        return Path(path)
    return config_dir / path


def read_line_file(path):
    """Read a PCS or dilution line file (x=x0,x1 and y=y0,y1). Returns ((x0, x1), (y0, y1), type)."""
    values = {}
    with open(path) as f:
        for row in f:
            if '=' in row:
                key, value = row.strip().split('=', 1)
                values[key] = value
    x = [float(v) for v in values['x'].split(',')]
    y = [float(v) for v in values['y'].split(',')]
    return x, y, values.get('type')


def draw_roi(ax, roi, colour, label):
    x0, y0, x1, y1 = roi
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colour, linewidth=1.5))
    ax.text(x0, y0 - 4, label, color=colour, fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Draw a site's PCS lines and ROIs on a camera image")
    parser.add_argument('--config', required=True, help="PyCam processing config (e.g. the site template)")
    parser.add_argument('--image', required=True, help="a plume image, normally fltrA")
    parser.add_argument('--out', default='overlay.png', help="output PNG")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    image = np.array(Image.open(args.image)).astype(float)
    low, high = np.percentile(image, [1, 99])   # stretch contrast so the plume is visible

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray', vmin=low, vmax=high)
    ax.set_title(f"{Path(args.image).name}\n{config_path.name}", fontsize=9)

    for i, line_file in enumerate(config.get('pcs_lines', [])):
        x, y, line_type = read_line_file(find_file(line_file, config_path.parent))
        if line_type is None:
            # Same default as PyplisWorker: first line young, second line old
            line_type = 'young' if i == 0 else 'old' if i == 1 else ''
        ax.plot(x, y, color='lime', linewidth=2)
        ax.text(x[1], y[1], f" PCS {i + 1} {line_type}", color='lime', fontsize=9)

    # Dilution lines and the source point only matter when their methods are switched on
    dil_lines = config.get('dil_lines', []) if config.get('use_light_dilution') else []
    for i, line_file in enumerate(dil_lines):
        x, y, _ = read_line_file(find_file(line_file, config_path.parent))
        ax.plot(x, y, color='orange', linewidth=1.5, linestyle='--')
        ax.text(x[1], y[1], f" dil {i + 1}", color='orange', fontsize=8)

    if config.get('ambient_roi'):
        draw_roi(ax, config['ambient_roi'], 'cyan', 'background ROI')
    if config.get('roi_abs'):
        draw_roi(ax, config['roi_abs'], 'magenta', 'optical flow ROI')

    if 'centre_pix_x' in config:
        centre = (config['centre_pix_x'], config['centre_pix_y'])
        radius = max(float(config.get('fov_rad', 1)), 3)   # draw at least 3 px so it can be seen
        ax.add_patch(Circle(centre, radius, fill=False, edgecolor='red', linewidth=2))
        ax.text(centre[0] + 6, centre[1], 'spectrometer FoV', color='red', fontsize=8)

    # The source point is the start of the Nadeau plume speed line
    if config.get('flow_nadeau') and config.get('source_coords'):
        sx, sy = config['source_coords']
        ax.plot(sx, sy, marker='x', color='yellow', markersize=10)
        ax.text(sx + 6, sy, 'source (Nadeau)', color='yellow', fontsize=8)

    ax.set_xlabel(f"image size {image.shape[1]} x {image.shape[0]} px")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out} (image {image.shape[1]} x {image.shape[0]} px)")


if __name__ == '__main__':
    main()
