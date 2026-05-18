import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def _pad(im, target_h=None, target_w=None):
    """Pad image with white to reach (target_h, target_w); skips if already big enough."""
    h, w = im.shape[:2]
    th = target_h or h
    tw = target_w or w
    if th <= h and tw <= w:
        return im
    pad_top = (th - h) // 2
    pad_left = (tw - w) // 2
    fill = 1.0 if np.issubdtype(im.dtype, np.floating) else 255
    if im.ndim == 3:
        out = np.full((th, tw, im.shape[2]), fill, dtype=im.dtype)
    else:
        out = np.full((th, tw), fill, dtype=im.dtype)
    out[pad_top:pad_top + h, pad_left:pad_left + w] = im
    return out


def combine_images(image_paths, save_path, titles=None,
                    orientation="horizontal", size=6,
                    preserve_relative_size=False, dpi=150):
    """
    preserve_relative_size:
        False -> all panels scaled to the same primary dim (height for
                 horizontal, width for vertical). Small plots get blown up.
        True  -> images keep their natural pixel ratios; smaller ones are
                 padded with whitespace so the strip stays aligned.
    """
    assert orientation in ("horizontal", "vertical")
    n = len(image_paths)
    images = [mpimg.imread(str(p)) for p in image_paths]

    if preserve_relative_size:
        if orientation == "horizontal":
            max_h = max(im.shape[0] for im in images)
            images = [_pad(im, target_h=max_h) for im in images]
        else:
            max_w = max(im.shape[1] for im in images)
            images = [_pad(im, target_w=max_w) for im in images]

    aspects = [im.shape[1] / im.shape[0] for im in images]

    if orientation == "horizontal":
        widths = [a * size for a in aspects]
        fig, axes = plt.subplots(
            1, n, figsize=(sum(widths), size),
            gridspec_kw={"wspace": 0, "hspace": 0, "width_ratios": widths},
        )
    else:
        heights = [size / a for a in aspects]
        fig, axes = plt.subplots(
            n, 1, figsize=(size, sum(heights)),
            gridspec_kw={"wspace": 0, "hspace": 0, "height_ratios": heights},
        )

    if n == 1:
        axes = [axes]

    for ax, im in zip(axes, images):
        ax.imshow(im, aspect="auto")
        ax.axis("off")

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return save_path


image_paths = [
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\SPD_rank_1_softmax\causal_importances_upper_leaky_30000.png", #rank 1
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8\causal_importances_upper_leaky_30000.png",
]

combine_images(
    image_paths,
    save_path="causal_importances_ranks_combined.png",
    orientation="horizontal",
    size=6,
    dpi=150
)


image_paths = [
                r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4\w1_w2_input_hidden_output_translation.png",
                r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\w1_w2_input_hidden_output_translation.png",
                r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8\w1_w2_input_hidden_output_translation.png",
               ]

combine_images(
    image_paths,
    save_path="input_hidden_output_translation_ranks_combined.png",
    orientation="vertical",
    size=6,
    dpi=150
)


image_paths = [r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2\group_output_matrix_simplex.png",
                r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\group_output_matrix_simplex.png",
                r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8\group_output_matrix_simplex.png"
                ]

combine_images(
    image_paths,
    save_path="group_output_matrix_simplex_ranks_combined.png",
    orientation="horizontal",
    size=6,
    dpi=150
)