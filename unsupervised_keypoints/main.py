import os
import wandb
import numpy as np
import torch
import numpy as np
from unsupervised_keypoints.optimize_token import load_ldm
from unsupervised_keypoints.optimize import optimize_embedding
from unsupervised_keypoints.config_utils import setup_config, convert_config_to_args

from unsupervised_keypoints.keypoint_regressor import (
    find_best_indices,
    precompute_all_keypoints,
    return_regressor,
    return_regressor_visible,
    return_regressor_human36m,
)

from unsupervised_keypoints.eval import evaluate
from unsupervised_keypoints.visualize import visualize_attn_maps


# 配置管理 - 使用OmegaConf替代argparse
print("正在加载配置...")
config = setup_config()
print(f"已加载配置文件，数据集: {config.dataset.name}, 设备: {config.training.device}")

# 为了保持与现有代码的兼容性，将配置转换为args对象
args = convert_config_to_args(config)

ldm, controllers, num_gpus = load_ldm(args.device, args.model_type, feature_upsample_res=args.feature_upsample_res, my_token=args.my_token)

# if args.save_folder doesnt exist create it
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
    
# print number of gpus
print("Number of GPUs: ", torch.cuda.device_count())

if args.wandb:
    # start a wandb session
    wandb.init(project="attention_maps", name=args.wandb_name, config=vars(args))

# Stage 1: Optimize Embedding (runs unconditionally)
embedding = optimize_embedding(
    ldm,
    args,
    controllers,
    num_gpus,
)
torch.save(embedding, os.path.join(args.save_folder, "embedding.pt"))
    
# Stage 2: Find Best Indices (runs unconditionally)
indices = find_best_indices(
    ldm,
    embedding,
    args,
    controllers,
    num_gpus,
)
torch.save(indices, os.path.join(args.save_folder, "indices.pt"))
    
if args.visualize:
    # Visualize embeddings after finding indices
    visualize_attn_maps(
        ldm,
        embedding,
        indices,
        args,
        controllers,
        num_gpus,
        # regressor is not available yet for the first visualization
    )

# Check for custom dataset before precomputation
if args.dataset_name == "custom":
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
        args,
        controllers,
        num_gpus,
    )

    torch.save(source_kpts, os.path.join(args.save_folder, "source_keypoints.pt"))
    torch.save(target_kpts, os.path.join(args.save_folder, "target_keypoints.pt"))
    if visible is not None: # visible can be None
        torch.save(visible, os.path.join(args.save_folder, "visible.pt"))

    # Stage 4: Train Regressor (runs if not custom dataset)
    if args.evaluation_method == "visible" or args.evaluation_method == "mean_average_error":
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
    elif args.evaluation_method == "orientation_invariant":
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
    torch.save(regressor, os.path.join(args.save_folder, "regressor.pt"))

    if args.visualize:
        # Visualize with regressor (runs if not custom dataset and visualize is true)
        visualize_attn_maps(
            ldm,
            embedding,
            indices,
            args,
            controllers,
            num_gpus,
            regressor=regressor.to(args.device),
        )

    # Stage 5: Evaluate (runs if not custom dataset)
    evaluate(
        ldm,
        embedding,
        indices,
        regressor.to(args.device),
        args,
        controllers,
        num_gpus,
    )
