import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import ceil
from pathlib import Path


def _pad(im, target_h=None, target_w=None):
    """Pad image with white to reach (target_h, target_w); skips if already big enough."""
    h, w = im.shape[:2]
    th = max(target_h or h, h)
    tw = max(target_w or w, w)
    if th == h and tw == w:
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
                   ncols=None, nrows=None, size=6,
                   preserve_relative_size=False, dpi=150):
    """
    Arrange images in a grid.

    ncols / nrows:
        Specify the grid shape. Give just one and the other is computed from
        the number of images; give both for an explicit layout; give neither
        and all images go in a single row.
            ncols=2 with 4 images -> 2x2 grid
            ncols=n               -> single row   (old "horizontal")
            nrows=n               -> single column (old "vertical")
        Images fill the grid row by row (left to right, top to bottom).

    preserve_relative_size:
        False -> every cell is the same size; each image is stretched to fill
                 its cell. No distortion when all images share an aspect ratio.
        True  -> images keep their natural pixel ratios; all are padded with
                 whitespace to a common size so nothing is rescaled.
    """
    images = [mpimg.imread(str(p)) for p in image_paths]
    n = len(images)

    # ---- work out the grid shape -------------------------------------------
    if ncols is None and nrows is None:
        ncols, nrows = n, 1
    elif ncols is None:
        ncols = ceil(n / nrows)
    elif nrows is None:
        nrows = ceil(n / ncols)

    # ---- normalise sizes / pick a uniform cell aspect ratio ----------------
    if preserve_relative_size:
        max_h = max(im.shape[0] for im in images)
        max_w = max(im.shape[1] for im in images)
        images = [_pad(im, target_h=max_h, target_w=max_w) for im in images]
        cell_aspect = max_w / max_h
    else:
        cell_aspect = float(np.mean([im.shape[1] / im.shape[0] for im in images]))

    # ---- build the figure ---------------------------------------------------
    cell_w = size * cell_aspect
    cell_h = size
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell_w, nrows * cell_h),
        gridspec_kw={"wspace": 0, "hspace": 0},
        squeeze=False,
    )
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i], aspect="auto")
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])
        ax.axis("off")  # also hides the empty trailing cells

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return save_path


#Complete minimality sweep image from [0.1 - 1e-6]

image_paths = [
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.1\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.01\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.001\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.0001\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.00001\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.000001\causal_importances_upper_leaky_30000.png"
]

combine_images(
    image_paths,
    save_path=r"C:\Users\Knud\uni\spd\thesis\written_product\Images\Softmax_minimality_sweep_full_1e1_to_1e-6.png",
    ncols=len(image_paths) // 2,
    size=6,
    dpi=350,
)

#Condensed sweep from 1e-3 to 1e-6
image_paths = [
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.001\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.00001\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.000001\causal_importances_upper_leaky_30000.png"
]

combine_images(
    image_paths,
    save_path=r"C:\Users\Knud\uni\spd\thesis\written_product\Images\Softmax_minimality_sweep_condensed_1e-3_to_1e-6.png",
    ncols=len(image_paths),
    size=6,
    dpi=350,
)


#full rank sweep at best minimality, #1e-3
image_paths = [
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_1\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_3\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_6\causal_importances_upper_leaky_30000.png",
]

combine_images(
    image_paths,
    save_path=r"C:\Users\Knud\uni\spd\thesis\written_product\Images\Softmax_causal_importances_ranking_sweep_minimality_1e-e3_rank_1_to_8.png",
    ncols=len(image_paths) // 2,
    size=6,
    dpi=350,
)

#PArtial rank sweep at best minimality, #1e-3
image_paths = [
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_1\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_3\causal_importances_upper_leaky_30000.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\causal_importances_upper_leaky_30000.png",
]

combine_images(
    image_paths,
    save_path=r"C:\Users\Knud\uni\spd\thesis\written_product\Images\Softmax_causal_importances_ranking_sweep_minimality_1e-e3_rank_1_3_5.png",
    ncols=len(image_paths),
    size=6,
    dpi=350,
)

#Full io sweep over minimalities
image_paths = [
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.1\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.01\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.001\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.0001\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.00001\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.000001\io_routing_chain.png",
]
combine_images(
    image_paths,
    save_path=r"C:\Users\Knud\uni\spd\thesis\written_product\Images\Softmax_minimality_io_sweep_full_1e1_to_1e-6.png",
    ncols=2,          # 2x2
    size=6,
    dpi=350,
)

#Full io sweep over ranks
image_paths = [
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_1\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_3\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_6\io_routing_chain.png",
]
combine_images(
    image_paths,
    save_path=r"C:\Users\Knud\uni\spd\thesis\written_product\Images\Softmax_rank_io_sweep_full_rank_1_to_8.png",
    ncols=2,          # 2x2
    size=6,
    dpi=350,
)

#Combined plots
image_paths = [
    #SPD
    r"C:\Users\Knud\uni\spd_original\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.001\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd_original\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.00001\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd_original\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.000001\io_routing_chain.png",
    #Softmax
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.001\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.00001\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.000001\io_routing_chain.png",
]
combine_images(
    image_paths,dpi=350,
    save_path=r"C:\Users\Knud\uni\spd\thesis\written_product\Images\Io_routing_chain_for_spd_and_softmax_combined_minimality_1e-3_and_1e-5.png",
    ncols=2,          # 2x2
    size=6,
)



#Combined io sweep at rank 3,5 for spd and softmax together
image_paths = [
    #SPD
    r"C:\Users\Knud\uni\spd_original\spd\experiments\toy_model_of_geometry\out\rank_sweep_2\rank_1\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd_original\spd\experiments\toy_model_of_geometry\out\rank_sweep_2\rank_3\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd_original\spd\experiments\toy_model_of_geometry\out\rank_sweep_2\rank_5\io_routing_chain.png",
    #Softmax
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_1\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_3\io_routing_chain.png",
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\io_routing_chain.png",
]

combine_images(
    image_paths,
    save_path=r"C:\Users\Knud\uni\spd\thesis\written_product\Images\Io_routing_chain_combined_rank_3_and_5.png",
    ncols=2,     
    size=8,
    dpi=350,
)
print("Combined image saved.")
