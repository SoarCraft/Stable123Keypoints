from src.config_utils import Config
import os
import torch
from tqdm import tqdm
from datasets.celeba import CelebA
from datasets import custom_images
from datasets import cub
from datasets import cub_parts
from datasets import taichi
from datasets import human36m
from datasets import unaligned_human36m
from datasets import deepfashion
from src.eval import run_image_with_context_augmented
from src.eval import pixel_from_weighted_avg, find_max_pixel
import matplotlib.pyplot as plt


def save_grid(maps, imgs, name, img_size=(512, 512), dpi=50, quality=85):
    """
    There are 10 maps of shape [32, 32]
    There are 10 imgs of shape [3, 512, 512]
    Saves as a single image with matplotlib with 2 rows and 10 columns
    Updated to have smaller borders between images and the edge.
    DPI is reduced to decrease file size.
    JPEG quality can be adjusted to trade off quality for file size.
    """

    # Calculate figure size to maintain aspect ratio
    fig_width = img_size[1] * 10  # total width for 10 images side by side
    fig_height = img_size[0] * 2  # total height for 2 images on top of each other
    fig_size = (fig_width / 100, fig_height / 100)  # scale down to a manageable figure size

    fig, axs = plt.subplots(2, 10, figsize=fig_size, gridspec_kw={'wspace':0.05, 'hspace':0.05})

    for i in range(10):
        axs[0, i].imshow(imgs[i].numpy().transpose(1, 2, 0))
        axs[1, i].imshow(imgs[i].numpy().transpose(1, 2, 0))
        normalized_map = maps[i] - torch.min(maps[i])
        normalized_map = normalized_map / torch.max(normalized_map)
        axs[1, i].imshow(normalized_map, alpha=0.7)

    # Remove axis and adjust subplot parameters
    for ax in axs.flatten():
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save as JPEG with reduced DPI and specified quality
    plt.savefig(name, format='jpg', bbox_inches='tight', pad_inches=0, dpi=dpi, pil_kwargs={'quality': quality})

    plt.close()


def plot_point_single(img, points, name):
    """
    Displays corresponding points on the image with white outline around plotted numbers.
    The numbers themselves retain their original color.
    points shape is [num_people, num_points, 2]
    """
    num_people, num_points, _ = points.shape

    # Get the default color cycle from Matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img.numpy().transpose(1, 2, 0))

    for i in range(num_people):
        for j in range(num_points):
            # Choose color based on j, cycling through the default color cycle
            color = colors[j % len(colors)]
            x, y = points[i, j, 1] * 512, points[i, j, 0] * 512
            # Plot the original color on top
            ax.scatter(x, y, color=color, marker=f"${j}$", s=300)

    ax.axis("off")  # Remove axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove border

    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)


def plot_point_correspondences(imgs, points, name, height = 11, width = 9):
    """
    Displays corresponding points per image
    len(imgs) = num_images
    points shape is [num_images, num_points, 2]
    """

    num_images, num_points, _ = points.shape

    fig, axs = plt.subplots(height, width, figsize=(2 * width, 2 * height))
    axs = axs.ravel()  # Flatten the 2D array of axes to easily iterate over it

    for i in range(height*width):
        axs[i].imshow(imgs[i].numpy().transpose(1, 2, 0))

        for j in range(num_points):
            # plot the points each as a different type of marker
            axs[i].scatter(
                points[i, j, 1] * 512.0, points[i, j, 0] * 512.0, marker=f"${j}$"
            )

    # remove axis and handle any unused subplots
    for i, ax in enumerate(axs):
        if i >= num_images:
            ax.axis("off")  # Hide unused subplots
        else:
            ax.axis("off")  # Remove axis from used subplots

    # Adjust subplot parameters to reduce space between images and border space
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    # increase the resolution of the plot
    plt.savefig(name, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


@torch.no_grad()
def visualize_attn_maps(
    ldm,
    context,
    indices,
    config: Config,
    controllers,
    num_gpus,
    regressor=None,
    from_where=["down_cross", "mid_cross", "up_cross"],
    height = 11,
    width = 9,
):
    if config.dataset_name == "celeba_aligned":
        dataset = CelebA(split="test", dataset_loc=config.dataset_loc)
    elif config.dataset_name == "celeba_wild":
        dataset = CelebA(split="test", dataset_loc=config.dataset_loc, align = False)
    elif config.dataset_name == "cub_aligned":
        dataset = cub.TestSet(data_root=config.dataset_loc, image_size=512)
    elif config.dataset_name == "cub_001":
        dataset = cub_parts.CUBDataset(dataset_root=config.dataset_loc, split="test", single_class=1)
    elif config.dataset_name == "cub_002":
        dataset = cub_parts.CUBDataset(dataset_root=config.dataset_loc, split="test", single_class=2)
    elif config.dataset_name == "cub_003":
        dataset = cub_parts.CUBDataset(dataset_root=config.dataset_loc, split="test", single_class=3)
    elif config.dataset_name == "cub_all":
        dataset = cub_parts.CUBDataset(dataset_root=config.dataset_loc, split="test")
    elif config.dataset_name == "taichi":
        dataset = taichi.TestSet(data_root=config.dataset_loc, image_size=512)
    elif config.dataset_name == "human3.6m":
        dataset = human36m.TestSet(data_root=config.dataset_loc, validation=config.validation)
    elif config.dataset_name == "unaligned_human3.6m":
        dataset = unaligned_human36m.TestSet(data_root=config.dataset_loc, image_size=512)
    elif config.dataset_name == "deepfashion":
        dataset = deepfashion.TestSet(data_root=config.dataset_loc, image_size=512)
    elif config.dataset_name == "custom":
        dataset = custom_images.CustomDataset(data_root=config.dataset_loc, image_size=512)
    else:
        raise NotImplementedError

    imgs = []
    maps = []
    gt_kpts = []
    
    # random permute the dataset
    randperm = torch.randperm(len(dataset))
    
    for i in tqdm(range(height * width)):
        batch = dataset[randperm[i%len(dataset)].item()]

        img = batch["img"]

        _gt_kpts = batch["kpts"] 
        gt_kpts.append(_gt_kpts)
        imgs.append(img.cpu())

        map_out = run_image_with_context_augmented(
            ldm,
            img,
            context,
            indices.cpu(),
            device=config.device,
            from_where=from_where,
            layers=config.layers,
            noise_level=config.noise_level,
            augment_degrees=config.augment_degrees,
            augment_scale=config.augment_scale,
            augment_translate=config.augment_translate,
            augmentation_iterations=config.augmentation_iterations,
            visualize=(i==0 and config.visualize),
            controllers=controllers,
            num_gpus=num_gpus,
            save_folder=config.save_folder,
        )

        maps.append(map_out.cpu())
    maps = torch.stack(maps)
    gt_kpts = torch.stack(gt_kpts)

    if config.max_loc_strategy == "argmax":
        points = find_max_pixel(maps.view(height * width * config.top_k, 512, 512)) / 512.0
    else:
        points = pixel_from_weighted_avg(maps.view(height * width * config.top_k, 512, 512)) / 512.0
    points = points.reshape(height * width, config.top_k, 2)

    plot_point_correspondences(
        imgs, points.cpu(), os.path.join(config.save_folder, "src.pdf"), height, width
    )

    for i in range(config.top_k):
        save_grid(
            maps[:, i].cpu(), imgs, os.path.join(config.save_folder, f"keypoint_{i:03d}.png")
        )

    if regressor is not None:
        est_points = ((points.view(height * width, -1)-0.5) @ regressor)+0.5

        plot_point_correspondences(
            imgs,
            est_points.view(height * width, -1, 2).cpu(),
            os.path.join(config.save_folder, "estimated_keypoints.pdf"),
            height,
            width,
        )

        plot_point_correspondences(
            imgs, gt_kpts, os.path.join(config.save_folder, "gt_keypoints.pdf"), height, width
        )
