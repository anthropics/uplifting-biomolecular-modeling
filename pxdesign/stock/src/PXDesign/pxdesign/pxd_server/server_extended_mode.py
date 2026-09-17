import argparse
import textwrap

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from server_constant import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Error")


def plot(args):
    out_ptx_sr = args.out_ptx_sr
    length = args.length
    use_temp_model = args.use_temp_model

    out_af2_sr = args.out_af2_sr

    if out_af2_sr > 40:
        out_af2_sr = 40
    if out_af2_sr < 5:
        out_af2_sr = 5
    if out_ptx_sr > 40:
        out_ptx_sr = 40
    if out_ptx_sr < 5:
        out_ptx_sr = 5

    length = args.length
    hard_afsr_cutoff = args.hard_afsr_cutoff

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
        0.472,
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
        0.081,
        0.340,
        hard,
        wrap=True,
        ha="center",
        multialignment="left",
        fontproperties=prop,
        fontsize=11,
        color="#086BFFE8",
    )
    plt.figtext(
        0.522,
        0.340,
        hard,
        wrap=True,
        ha="center",
        multialignment="left",
        fontproperties=prop,
        fontsize=11,
        color="#086BFFE8",
    )
    plt.figtext(
        0.912,
        0.340,
        easy,
        wrap=True,
        ha="center",
        multialignment="left",
        fontproperties=prop,
        fontsize=11,
        color="#086BFFE8",
    )

    AF2_SR = get_information(length)["AF2_SR"]
    AF2_SR = np.array(AF2_SR)
    AF2_SR[AF2_SR > 40] = 40
    AF2_SR[AF2_SR < 5] = 5

    length_count = get_information(length)["count"]

    ax1 = fig.add_axes([0.075, 0.2, 0.40, 0.3])
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
    if out_af2_sr > hard_afsr_cutoff[-1]:
        color_sr = difficulty_to_color["easy"]
    elif out_af2_sr > hard_afsr_cutoff[0] and out_af2_sr <= hard_afsr_cutoff[-1]:
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
    ax1.set_xticks([5, 10, 20, 30, 40])
    ax1.set_xticklabels([r"$\leq 5$%", "10%", "20%", "30%", r"$\geq 40$%"])

    ax2 = fig.add_axes([0.5175, 0.2, 0.40, 0.3])
    if_template_model = " (Template Model) " if use_temp_model else " "
    ax2.set_title(
        f"Protenix-basic Passing Rate  {if_template_model}({length_count})",
        fontproperties=prop,
        fontsize=10,
    )
    ax2.hlines(y=0, xmin=3, xmax=42.0, color="black", linewidth=1.5, alpha=0.7)
    ptx_SR = (
        get_information(length)["ptx_temp_sr"]
        if use_temp_model
        else get_information(length)["ptx_sr"]
    )

    ptx_SR = np.array(ptx_SR)
    ptx_SR[ptx_SR < 5] = 5
    ptx_SR[ptx_SR > 40] = 40

    placed_positions = []
    y_add = 0.01
    for x, label in zip(ptx_SR, labels):
        ax2.plot(x, 0, "o", color="dimgray", markersize=6, alpha=0.7)
        if label:
            for px, py in placed_positions:
                if abs(x - px) < 0.5:
                    x = px + 0.7
            placed_positions.append((x, y_add))
            ax2.text(
                x,
                y_add,
                label,
                fontsize=5,
                rotation=70,
                ha="center",
                va="bottom",
                fontproperties=prop,
            )

    ax2.plot(out_ptx_sr, 0, "o", color=color_sr, markersize=6)
    ax2.annotate(
        description,
        xy=(out_ptx_sr, 0),
        xytext=(out_ptx_sr, -0.04),
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
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    x_tick_labels = ax2.get_xticklabels()
    for label in x_tick_labels:
        label.set_fontproperties(prop)
        label.set_fontsize(8)
    ax2.set_yticks([])
    ax2.set_xticks(
        [
            5,
            10,
            20,
            30,
            40,
        ]
    )
    ax2.set_xticklabels([r"$\leq 5$%", "10%", "20%", "30%", r"$\geq 40$%"])

    # plt.savefig(args.save_path, dpi=800, bbox_inches="tight")
    plt.savefig(args.save_path, dpi=400)
    print(f"Save difficulty fig to {args.save_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_af2_sr", type=float, default=20, help="af2 sr")
    parser.add_argument("--out_ptx_sr", type=float, default=22, help="ptx iptm")
    parser.add_argument(
        "--use_temp_model",
        type=str2bool,
        default=False,
        help="use large model or template model",
    )
    parser.add_argument("--length", type=int, default=100, help="binder length")
    parser.add_argument(
        "--hard_AF2_ipae_cutoff",
        type=float,
        default=[10, 15],
        help="hard case ipae cutoff (<10, 10-15, >15)",
    )

    parser.add_argument(
        "--hard_ptx_iptm_template_cutoff",
        type=float,
        default=[0.6, 0.3],
        help="iptm cutoff (0.6-1.0, 0.3-0.6, <0.3)",
    )
    parser.add_argument(
        "--hard_afsr_cutoff",
        type=float,
        default=[15, 5],
        help="hard case sr cutoff (0-5%, 5-15%, >15%)",
    )
    parser.add_argument("--save_path", type=str, default="./server_extended_mode.png")

    args = parser.parse_args()

    plot(args)
