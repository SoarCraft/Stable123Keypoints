import time
import torch
from tqdm import tqdm
from src import ptp_utils
from src import eval
import torch.nn.functional as F
from datasets.celeba import CelebA
from datasets import custom_images
from datasets import cub_aligned
from datasets import cub_parts
from datasets import taichi
from datasets import human36m
from datasets import unaligned_human36m
from datasets import deepfashion
from src import optimize_token
import wandb
from src.invertable_transform import RandomAffineWithInverse
from src.config_utils import Config


def collect_maps(
    controller,
    from_where=["up_cross"],
    upsample_res=512,
    layers=[0, 1, 2, 3],
    indices=None,
    device="cuda",
):
    """
    returns the bilinearly upsampled attention map of size upsample_res x upsample_res for the first word in the prompt
    """

    attention_maps = controller.step_store['attn']

    attention_maps_list = []

    layer_overall = -1

    for layer in range(len(attention_maps)):
        layer_overall += 1

        if layer_overall not in layers:
            continue

        data = attention_maps[layer]
        
        data = data.to(device)

        data = data.reshape(
            data.shape[0], int(data.shape[1] ** 0.5), int(data.shape[1] ** 0.5), data.shape[2]
        )
        
        # import ipdb; ipdb.set_trace()

        if indices is not None:
            data = data[:, :, :, indices]

        data = data.permute(0, 3, 1, 2)

        if upsample_res != -1 and data.shape[1] ** 0.5 != upsample_res:
            # bilinearly upsample the image to attn_sizexattn_size
            data = F.interpolate(
                data,
                size=(upsample_res, upsample_res),
                mode="bilinear",
                align_corners=False,
            )

        attention_maps_list.append(data)


    attention_maps_list = torch.stack(attention_maps_list, dim=0).mean(dim=(0, 1))

    controller.reset()

    return attention_maps_list


def create_gaussian_kernel(size: int, sigma: float):
    """
    Create a 2D Gaussian kernel of given size and sigma.

    Args:
        size (int): The size (width and height) of the kernel. Should be odd.
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        Tensor: A 2D tensor representing the Gaussian kernel.
    """
    assert size % 2 == 1, "Size must be odd"
    center = size // 2

    x = torch.arange(0, size, dtype=torch.float32)
    y = torch.arange(0, size, dtype=torch.float32)
    x, y = torch.meshgrid(x - center, y - center)

    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel


def equivariance_loss(embeddings_initial, embeddings_transformed, transform, index):
    # untransform the embeddings_transformed
    embeddings_initial_prime = transform.inverse(embeddings_transformed)[index]

    loss = F.mse_loss(embeddings_initial, embeddings_initial_prime)

    return loss


def sharpening_loss(attn_map, sigma=1.0, temperature=1e1, device="cuda", num_subjects = 1):
    pos = eval.find_k_max_pixels(attn_map, num=num_subjects)/attn_map.shape[-1]

    loss = find_gaussian_loss_at_point(
        attn_map,
        pos,
        sigma=sigma,
        temperature=temperature,
        device=device,
        num_subjects=num_subjects,
    )

    return loss


def find_gaussian_loss_at_point(
    attn_map, pos, sigma=1.0, temperature=1e-1, device="cuda", indices=None, num_subjects=1
):
    """
    pos is a location between 0 and 1
    """

    # attn_map is of shape (T, H, W)
    T, H, W = attn_map.shape

    # Create Gaussian circle at the given position
    target = optimize_token.gaussian_circles(
        pos, size=H, sigma=sigma, device=attn_map.device
    )  # Assuming H and W are the same
    target = target.to(attn_map.device)

    # possibly select a subset of indices
    if indices is not None:
        attn_map = attn_map[indices]
        target = target[indices]

    # Compute loss
    loss = F.mse_loss(attn_map, target)

    return loss


def optimize_embedding(
    ldm,
    config: Config,
    controllers,
    num_gpus,
    context=None,
    from_where=["down_cross", "mid_cross", "up_cross"],
):
    
    if config.dataset_name == "celeba_aligned":
        dataset = CelebA(split="train", dataset_loc=config.dataset_loc, max_len=config.max_len)
    elif config.dataset_name == "celeba_wild":
        dataset = CelebA(split="train", dataset_loc=config.dataset_loc, align = False, max_len=config.max_len)
    elif config.dataset_name == "cub_aligned":
        dataset = cub_aligned.TrainSet(data_root=config.dataset_loc, image_size=512)
    elif config.dataset_name == "cub_001":
        dataset = cub_parts.CUBDataset(dataset_root=config.dataset_loc, split="train", single_class=1)
    elif config.dataset_name == "cub_002":
        dataset = cub_parts.CUBDataset(dataset_root=config.dataset_loc, split="train", single_class=2)
    elif config.dataset_name == "cub_003":
        dataset = cub_parts.CUBDataset(dataset_root=config.dataset_loc, split="train", single_class=3)
    elif config.dataset_name == "cub_all":
        dataset = cub_parts.CUBDataset(dataset_root=config.dataset_loc, split="train")
    elif config.dataset_name == "taichi":
        dataset = taichi.TrainSet(data_root=config.dataset_loc, image_size=512)
    elif config.dataset_name == "human3.6m":
        dataset = human36m.TrainSet(data_root=config.dataset_loc, validation=config.validation)
    elif config.dataset_name == "unaligned_human3.6m":
        dataset = unaligned_human36m.TrainSet(data_root=config.dataset_loc, image_size=512)
    elif config.dataset_name == "deepfashion":
        dataset = deepfashion.TrainSet(data_root=config.dataset_loc, image_size=512)
    elif config.dataset_name == "custom":
        dataset = custom_images.CustomDataset(data_root=config.dataset_loc, image_size=512)
    else:
        raise NotImplementedError


    invertible_transform = RandomAffineWithInverse(
        degrees=config.augment_degrees,
        scale=config.augment_scale,
        translate=config.augment_translate,
    )

    # every iteration return image, pixel_loc

    if context is None:
        context = ptp_utils.init_random_noise(config.device, num_words=config.num_tokens)

    context.requires_grad = True

    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.AdamW([context], lr=config.lr, weight_decay=1e-4)

    # time the optimization
    start = time.time()
    it_start = time.time()

    running_equivariance_attn_loss = 0
    running_sharpening_loss = 0
    running_total_loss = 0
    
    # create dataloader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_gpus, shuffle=True, drop_last=True)

    dataloader_iter = iter(dataloader)
    
    for iteration in tqdm(range(int(int(config.num_steps)*(config.batch_size//num_gpus)))):
        
        try:
            mini_batch = next(dataloader_iter)
        except StopIteration:  # Explicitly catch StopIteration
            dataloader_iter = iter(dataloader)
            mini_batch = next(dataloader_iter)

        image = mini_batch["img"]

        attn_maps = ptp_utils.run_and_find_attn(
            ldm,
            image,
            context,
            layers=config.layers,
            noise_level=config.noise_level,
            from_where=from_where,
            upsample_res=-1,
            device=config.device,
            controllers=controllers,
        )

        transformed_img = invertible_transform(image)

        attention_maps_transformed = ptp_utils.run_and_find_attn(
            ldm,
            transformed_img,
            context,
            layers=config.layers,
            noise_level=config.noise_level,
            from_where=from_where,
            upsample_res=-1,
            device=config.device,
            controllers=controllers,
        )
        
        _sharpening_loss = []
        _loss_equivariance_attn = []
        
        for index, attn_map, attention_map_transformed in zip(torch.arange(num_gpus), attn_maps, attention_maps_transformed):

            if config.top_k_strategy == "entropy":
                top_embedding_indices = ptp_utils.entropy_sort(
                    attn_map, config.furthest_point_num_samples,
                )
            elif config.top_k_strategy == "gaussian":
                top_embedding_indices = ptp_utils.find_top_k_gaussian(
                    attn_map, config.furthest_point_num_samples, sigma=config.sigma, num_subjects = config.num_subjects
                )
            elif config.top_k_strategy == "consistent":
                top_embedding_indices = torch.arange(config.furthest_point_num_samples)
            else:
                raise NotImplementedError
            
            top_embedding_indices = ptp_utils.furthest_point_sampling(attention_map_transformed, config.top_k, top_embedding_indices)

            _sharpening_loss.append(sharpening_loss(attn_map[top_embedding_indices], device=config.device, sigma=config.sigma, num_subjects = config.num_subjects))

            _loss_equivariance_attn.append(equivariance_loss(
                attn_map[top_embedding_indices], attention_map_transformed[top_embedding_indices][None].repeat(num_gpus, 1, 1, 1), invertible_transform, index
            ))
        
        _sharpening_loss = torch.stack([loss.to('cuda:0') for loss in _sharpening_loss]).mean()
        _loss_equivariance_attn = torch.stack([loss.to('cuda:0') for loss in _loss_equivariance_attn]).mean()

        loss = (
            + _loss_equivariance_attn * config.equivariance_attn_loss_weight
            + _sharpening_loss * config.sharpening_loss_weight
        )

        running_equivariance_attn_loss += _loss_equivariance_attn / (config.batch_size//num_gpus) * config.equivariance_attn_loss_weight
        running_sharpening_loss += _sharpening_loss / (config.batch_size//num_gpus) * config.sharpening_loss_weight
        running_total_loss += loss / (config.batch_size//num_gpus)

        loss = loss / (config.batch_size//num_gpus)

        loss.backward()
        
        if (iteration + 1) % (config.batch_size//num_gpus) == 0:
            optimizer.step()
            optimizer.zero_grad()

            if config.wandb_enabled:
                wandb.log(
                    {
                        "loss": running_total_loss.item(),
                        "running_equivariance_attn_loss": running_equivariance_attn_loss.item(),
                        "running_sharpening_loss": running_sharpening_loss.item(),
                        "iteration time": time.time() - it_start,
                    }
                )
            else:
                print(
                    f"loss: {loss.item()}, \
                    _loss_equivariance_attn: {running_equivariance_attn_loss.item()} \
                    sharpening_loss: {running_sharpening_loss.item()},  \
                    running_total_loss: {running_total_loss.item()}, \
                    iteration time: {time.time() - it_start}"
                )
            running_equivariance_attn_loss = 0
            running_sharpening_loss = 0
            running_total_loss = 0
            
            it_start = time.time()

    print(f"optimization took {time.time() - start} seconds")

    return context.detach()
