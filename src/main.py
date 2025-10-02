import os
import wandb
import numpy as np
import torch
import numpy as np
from src.optimize_token import load_model
from src.optimize import optimize_embedding
from src.config_utils import setup_config

from src.keypoint_regressor import (
    find_best_indices,
    precompute_all_keypoints,
    return_regressor,
    return_regressor_visible,
    return_regressor_human36m,
)

from src.eval import evaluate
from src.visualize import visualize_attn_maps


print("正在加载配置...")
config = setup_config()
print(f"已加载配置文件，数据集: {config.dataset_name}, 设备: {config.device}")

ldm, controllers, num_gpus = load_model(config.device, config.model_type, feature_upsample_res=config.feature_upsample_res, my_token=config.my_token)

# if config.save_folder doesnt exist create it
if not os.path.exists(config.save_folder):
    os.makedirs(config.save_folder)

if config.wandb_enabled:
    # start a wandb session
    wandb_init_kwargs = {
        "project": config.wandb_project,
        "name": config.wandb_name
    }
    if config.wandb_entity:
        wandb_init_kwargs["entity"] = config.wandb_entity
    wandb.init(**wandb_init_kwargs)

# Stage 1: Optimize Embedding (runs unconditionally)
embedding = optimize_embedding(
    ldm,
    config,
    controllers,
    num_gpus,
)
torch.save(embedding, os.path.join(config.save_folder, "embedding.pt"))
    
# Stage 2: Find Best Indices (runs unconditionally)
indices = find_best_indices(
    ldm,
    embedding,
    config,
    controllers,
    num_gpus,
)
torch.save(indices, os.path.join(config.save_folder, "indices.pt"))
    
if config.visualize:
    # Visualize embeddings after finding indices
    visualize_attn_maps(
        ldm,
        embedding,
        indices,
        config,
        controllers,
        num_gpus,
        # regressor is not available yet for the first visualization
    )

# Check for custom dataset before precomputation
if config.dataset_name == "custom":
    print("Dataset is 'custom'. Skipping precomputation, regressor training, and evaluation stages.")
    # If you want to exit completely after visualization for custom datasets:
    # import sys
    # sys.exit(0)
else:
    # Stage 3: Precompute Keypoints (runs if not custom dataset)
    source_kpts, target_kpts, visible = precompute_all_keypoints(
        ldm,
        embedding,
        indices,
        config,
        controllers,
        num_gpus,
    )

    torch.save(source_kpts, os.path.join(config.save_folder, "source_keypoints.pt"))
    torch.save(target_kpts, os.path.join(config.save_folder, "target_keypoints.pt"))
    if visible is not None: # visible can be None
        torch.save(visible, os.path.join(config.save_folder, "visible.pt"))

    # Stage 4: Train Regressor (runs if not custom dataset)
    if config.evaluation_method == "visible" or config.evaluation_method == "mean_average_error":
        if visible is None:
            # If visible is None from precompute (e.g. custom dataset didn't yield it, though we stop before this for custom)
            # or if a dataset type simply doesn't provide visibility.
            # Create a dummy visible tensor full of ones if it's required by evaluation but not provided.
            # This part might need adjustment based on how precompute_all_keypoints handles visible for all dataset types.
            # For now, assuming target_kpts is available to infer shape.
            visible_reshaped = torch.ones_like(target_kpts).reshape(target_kpts.shape[0], target_kpts.shape[1] * 2).cpu().numpy().astype(np.float64)
        else:
            visible_reshaped = visible.unsqueeze(-1).repeat(1, 1, 2).reshape(visible.shape[0], visible.shape[1] * 2).cpu().numpy().astype(np.float64)

        regressor = return_regressor_visible( 
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1]*2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1]*2).astype(np.float64),
            visible_reshaped,
        )
    elif config.evaluation_method == "orientation_invariant":
        regressor = return_regressor_human36m( 
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1]*2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1]*2).astype(np.float64),
        )
    else:
        regressor = return_regressor( 
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1]*2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1]*2).astype(np.float64),
        )
    regressor = torch.tensor(regressor).to(torch.float32)
    torch.save(regressor, os.path.join(config.save_folder, "regressor.pt"))

    if config.visualize:
        # Visualize with regressor (runs if not custom dataset and visualize is true)
        visualize_attn_maps(
            ldm,
            embedding,
            indices,
            config,
            controllers,
            num_gpus,
            regressor=regressor.to(config.device),
        )

    # Stage 5: Evaluate (runs if not custom dataset)
    evaluate(
        ldm,
        embedding,
        indices,
        regressor.to(config.device),
        config,
        controllers,
        num_gpus,
    )
