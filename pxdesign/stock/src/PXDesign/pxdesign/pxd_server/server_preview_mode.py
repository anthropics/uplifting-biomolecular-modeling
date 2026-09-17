import argparse
import textwrap

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from server_constant import *


def plot(args):
    out_af2_ipae = args.out_af2_ipae
    out_af2_sr = args.out_af2_sr

    if out_af2_sr > 40:
        out_af2_sr = 40
    if out_af2_sr < 5:
        out_af2_sr = 5
    length = args.length
    hard_ipae_cutoff = args.hard_ipae_cutoff
    hard_sr_cutoff = args.hard_sr_cutoff

    label = "Average Top10% AlphaFold2-IG interface pAE"
    fig = plt.figure(figsize=(14, 2))
    wrapped = "\n".join(textwrap.wrap(text_preview, width=170))  # 每行20字符

    prop = fm.FontProperties(
        fname=os.path.join(os.path.dirname(font_path), "TimesNewRoman.ttf")
    )
    plt.figtext(
        0.5,
        0.750,
        wrapped,
        wrap=True,
        ha="center",
        multialignment="left",
        fontproperties=prop,
        fontsize=13,
        color="navy",
    )

    prop = fm.FontProperties(fname=font_path)
    easy = "easy"
    hard = "hard"
    plt.figtext(
        0.70,
        0.340,
        easy,
        wrap=True,
        ha="center",
        multialignment="left",
        fontproperties=prop,
        fontsize=11,
        color="#086BFFE8",
    )
    plt.figtext(
        0.311,
        0.340,
        hard,
        wrap=True,
        ha="center",
        multialignment="left",
        fontproperties=prop,
        fontsize=11,
        color="#086BFFE8",
    )

    AF2_ipae = get_information(length)["AF2_ipae"]
    AF2_ipae = np.array(AF2_ipae)
    AF2_ipae[AF2_ipae > 16.0] = 16.0

    AF2_SR = get_information(length)["AF2_SR"]
    AF2_SR = np.array(AF2_SR)
    AF2_SR[AF2_SR > 40] = 40
    AF2_SR[AF2_SR < 5] = 5
    length_count = get_information(length)["count"]

    ax1 = fig.add_axes([0.305, 0.2, 0.40, 0.3])
    ax1.set_title(
        f"AF2-IG-easy Passing Rate  ({length_count})",
        fontproperties=prop,
        fontsize=10,
    )
    ax1.hlines(y=0, xmin=3, xmax=42, color="black", linewidth=1.5, alpha=0.7)
    placed_positions = []
    y_add = 0.01

    for x, label in zip(AF2_SR, labels):
        ax1.plot(x, 0, "o", color="dimgray", markersize=6, alpha=0.7)
        if label:
            for px, py in placed_positions:
                if abs(x - px) < 0.5:
                    x = x + 0.7
            placed_positions.append((x, y_add))
            ax1.text(
                x,
                y_add,
                label,
                fontsize=5,
                rotation=70,
                ha="center",
                va="bottom",
                fontproperties=prop,
            )

    description = "Your job is here."
    if out_af2_sr > hard_sr_cutoff[-1]:
        color_sr = difficulty_to_color["easy"]
    elif out_af2_sr > hard_sr_cutoff[0] and out_af2_sr <= hard_sr_cutoff[-1]:
        color_sr = difficulty_to_color["hard"]
    else:
        color_sr = difficulty_to_color["very_hard"]

    ax1.plot(out_af2_sr, 0, "o", color=color_sr, markersize=6)
    ax1.annotate(
        description,
        xy=(out_af2_sr, 0),
        xytext=(out_af2_sr, -0.04),
        ha="center",
        color=color_sr,
        arrowprops=dict(
            arrowstyle="->",
            color=color_sr,
            connectionstyle="angle,angleA=90,angleB=180,rad=0",
        ),
        fontproperties=prop,
        fontsize=6,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    x_tick_labels = ax1.get_xticklabels()
    for label in x_tick_labels:
        label.set_fontproperties(prop)
        label.set_fontsize(8)
    ax1.set_yticks([])
    ax1.set_xticks(
        [
            5,
            10,
            20,
            30,
            40,
        ]
    )
    ax1.set_xticklabels([r"$\leq 5$%", "10%", "20%", "30%", r"$\geq 40$%"])

    # plt.savefig(args.save_path, dpi=800, bbox_inches="tight")
    plt.savefig(args.save_path, dpi=400)
    # plt.show()
    print(f"Save difficulty fig to {args.save_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_af2_ipae", type=float, default=14, help="output af2 ipae"
    )
    parser.add_argument("--out_af2_sr", type=float, default=30, help="output af2 sr")
    parser.add_argument("--length", type=int, default=87, help="binder length")
    parser.add_argument(
        "--hard_ipae_cutoff",
        type=float,
        default=[10, 15],
        help="hard case ipae cutoff (<10, 10-15, >15)",
    )
    parser.add_argument(
        "--hard_sr_cutoff",
        type=float,
        default=[15, 5],
        help="hard case sr cutoff (0-5%, 5-15%, >15%)",
    )
    parser.add_argument("--save_path", type=str, default="./server_preview_mode.png")

    args = parser.parse_args()
    plot(args)
