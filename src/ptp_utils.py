import torch.distributions as dist
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import abc
from src.eval import find_max_pixel, find_k_max_pixels
from src import optimize_token
from PIL import Image
from src.optimize import collect_maps
from diffusers.models.attention_processor import AttnProcessor2_0


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, dict, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, dict, is_cross: bool, place_in_unet: str):
        
        dict = self.forward(dict, is_cross, place_in_unet)
        
        return dict['attn']

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "attn": [],
        }

    def forward(self, dict, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 32**2:  # avoid memory overhead
        self.step_store["attn"].append(dict['attn']) 
        
        return dict

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()


def find_top_k_gaussian(attention_maps, top_k, sigma = 3, epsilon = 1e-5, num_subjects = 1):
    """
    attention_maps is of shape [batch_size, image_h, image_w]
    
    min_dist set to 0 becomes a simple top_k
    """
    
    device = attention_maps.device

    batch_size, image_h, image_w = attention_maps.shape

    max_pixel_locations = find_k_max_pixels(attention_maps, num=num_subjects)/image_h

    # Normalize the activation maps to represent probability distributions
    attention_maps_softmax = torch.softmax(attention_maps.view(batch_size, image_h * image_w)+epsilon, dim=-1)

    target = optimize_token.gaussian_circles(max_pixel_locations, size=image_h, sigma=sigma, device=attention_maps.device)
    
    target = target.reshape(batch_size, image_h * image_w)+epsilon
    target /= target.sum(dim=-1, keepdim=True)

    # sort the kl distances between attention_maps_softmax and target
    kl_distances = torch.sum(target * (torch.log(target) - torch.log(attention_maps_softmax)), dim=-1)
    # get the argsort for kl_distances
    kl_distances_argsort = torch.argsort(kl_distances, dim=-1, descending=False)
    
    return kl_distances_argsort[:top_k].detach().clone().to(device)


def furthest_point_sampling(attention_maps, top_k, top_initial_candidates):
    """
    attention_maps is of shape [batch_size, image_h, image_w]
    
    min_dist set to 0 becomes a simple top_k
    """
    
    device = attention_maps.device

    batch_size, image_h, image_w = attention_maps.shape

    # Assuming you have a function find_max_pixel to get the pixel locations
    max_pixel_locations = find_max_pixel(attention_maps)/image_h  # You'll need to define find_max_pixel

    # Find the furthest two points from top_initial_candidates
    max_dist = -1
    
    for i in range(len(top_initial_candidates)):
        for j in range(i+1, len(top_initial_candidates)):
            dist = torch.sqrt(torch.sum((max_pixel_locations[top_initial_candidates[i]] - max_pixel_locations[top_initial_candidates[j]])**2))
            if dist > max_dist:
                max_dist = dist
                furthest_pair = (top_initial_candidates[i].item(), top_initial_candidates[j].item())

    # Initialize the furthest point sampling with the furthest pair
    selected_indices = [furthest_pair[0], furthest_pair[1]]
    
    for _ in range(top_k - 2):
        max_min_dist = -1
        furthest_point = None
        
        for i in top_initial_candidates:
            if i.item() in selected_indices:
                continue
            
            this_min_dist = torch.min(torch.sqrt(torch.sum((max_pixel_locations[i] - torch.index_select(max_pixel_locations, 0, torch.tensor(selected_indices).to(device)))**2, dim=-1)))
            
            if this_min_dist > max_min_dist:
                max_min_dist = this_min_dist
                furthest_point = i.item()
        
        if furthest_point is not None:
            selected_indices.append(furthest_point)
    
    return torch.tensor(selected_indices).to(device)


def entropy_sort(attention_maps, top_k, min_dist=0.05):
    """
    attention_maps is of shape [batch_size, image_h, image_w]
    
    min_dist set to 0 becomes a simple top_k
    """
    
    device = attention_maps.device
    
    batch_size, image_h, image_w = attention_maps.shape
    
    max_pixel_locations = find_max_pixel(attention_maps)/image_h

    # Normalize the activation maps to represent probability distributions
    attention_maps_softmax = torch.softmax(attention_maps.view(batch_size, image_h * image_w), dim=-1)

    # Compute the entropy of each token
    entropy = dist.Categorical(probs=attention_maps_softmax).entropy()
    
    # find the argsort for entropy
    entropy_argsort = torch.argsort(entropy, dim=-1, descending=False)
    
    return entropy_argsort[:top_k]


def find_pred_noise(
    ldm,
    image,
    context,
    noise_level=-1,
    device="cuda",
):
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(0, 2, 3, 1).detach().cpu().numpy()

    with torch.no_grad():
        cond_lat = image2latent(ldm, image, device)
        
    timestep = ldm.scheduler.timesteps[noise_level]
    noise = torch.randn_like(cond_lat)

    noisy_latent = ldm.scheduler.add_noise(
        cond_lat, noise, timestep
    )
    
    with autocast(device):
        pred_noise = ldm.unet(noisy_latent, 
                              timestep.repeat(noisy_latent.shape[0]), 
                              encoder_hidden_states=context.repeat(noisy_latent.shape[0], 1, 1),
                              cross_attention_kwargs={"cond_lat": cond_lat})["sample"]
    
    return noise, pred_noise


def run_and_find_attn(
    ldm,
    image,
    context,
    noise_level=-1,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    upsample_res=32,
    indices=None,
    controllers=None,
):
    _, _ = find_pred_noise(
        ldm,
        image,
        context,
        noise_level=noise_level,
        device=device,
    )
    
    attention_maps=[]
    
    for controller in controllers:

        _attention_maps = collect_maps(
            controllers[controller],
            from_where=from_where,
            upsample_res=upsample_res,
            layers=layers,
            indices=indices,
            device=device,
        )
        
        attention_maps.append(_attention_maps)

        controllers[controller].reset()
        
    return attention_maps


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def image2latent(ldm, image, device):
    with torch.no_grad():
        if type(image) is Image:
            image = to_rgb_image(image)
            image = ldm.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
            image = torch.from_numpy(image)
            image = image.to(device)

        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            with autocast(device):
                if isinstance(ldm.vae, torch.nn.DataParallel):
                    latents = ldm.vae.module.encode(image)["latent_dist"].mean
                    latents = latents * ldm.vae.module.config.scaling_factor
                else:
                    latents = ldm.vae.encode(image)["latent_dist"].mean
                    latents = latents * ldm.vae.config.scaling_factor
            
    return latents


def register_attention_control(model, controller, feature_upsample_res=128):
    class ControlledAttnProcessor2_0(AttnProcessor2_0):
        def __init__(self, controller, place_in_unet, feature_upsample_res=128):
            super().__init__()
            self.controller = controller
            self.place_in_unet = place_in_unet
            self.feature_upsample_res = feature_upsample_res
            
        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
        ):
            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = F.softmax(attention_scores, dim=-1)
            
            if (
                is_cross and 
                hidden_states.shape[1] <= 32**2 and 
                len(self.controller.step_store["attn"]) < 3
            ):
                hidden_seq_len = hidden_states.shape[1]
                sqrt_seq_len = int(hidden_seq_len**0.5)
                
                if sqrt_seq_len * sqrt_seq_len == hidden_seq_len:
                    x_reshaped = hidden_states.reshape(
                        batch_size,
                        sqrt_seq_len,
                        sqrt_seq_len,
                        hidden_states.shape[-1],
                    ).permute(0, 3, 1, 2)
                    
                    x_reshaped = (
                        F.interpolate(
                            x_reshaped,
                            size=(self.feature_upsample_res, self.feature_upsample_res),
                            mode="bicubic",
                            align_corners=False,
                        )
                        .permute(0, 2, 3, 1)
                        .reshape(batch_size, -1, hidden_states.shape[-1])
                    )

                    q_upsampled = attn.to_q(x_reshaped)
                    q_upsampled = q_upsampled.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    attn_scores_upsampled = torch.matmul(q_upsampled, key.transpose(-2, -1)) * attn.scale
                    attn_probs_upsampled = F.softmax(attn_scores_upsampled, dim=-1)
                    
                    attention_probs = self.controller(
                        {"attn": attn_probs_upsampled}, 
                        is_cross, 
                        self.place_in_unet
                    )
                    
                    upsampled_output = torch.matmul(attention_probs, value)
                    
                    upsampled_output = upsampled_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                    upsampled_h = upsampled_w = self.feature_upsample_res
                    upsampled_output_reshaped = upsampled_output.reshape(
                        batch_size, upsampled_h, upsampled_w, -1
                    ).permute(0, 3, 1, 2)
                    
                    downsampled_output = F.interpolate(
                        upsampled_output_reshaped,
                        size=(sqrt_seq_len, sqrt_seq_len),
                        mode="bicubic",
                        align_corners=False,
                    ).permute(0, 2, 3, 1).reshape(batch_size, hidden_seq_len, -1)
                    
                    hidden_states = downsampled_output.view(batch_size, hidden_seq_len, attn.heads, head_dim).transpose(1, 2)
                else:
                    attention_probs = self.controller(
                        {"attn": attention_probs}, 
                        is_cross, 
                        self.place_in_unet
                    )
                    hidden_states = torch.matmul(attention_probs, value)
            else:
                hidden_states = torch.matmul(attention_probs, value)
            
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0
            self.step_store = {"attn": []}

    if controller is None:
        controller = DummyController()

    attn_procs = model.attn_processors
    new_attn_procs = {}
    
    cross_att_count = 0
    
    for name, processor in attn_procs.items():
        if "down_blocks" in name:
            place_in_unet = "down"
        elif "mid_block" in name:
            place_in_unet = "mid"  
        elif "up_blocks" in name:
            place_in_unet = "up"
        else:
            place_in_unet = "unknown"
        
        if "attn2" in name and "up_blocks" in name:
            new_attn_procs[name] = ControlledAttnProcessor2_0(
                controller, place_in_unet, feature_upsample_res
            )
            cross_att_count += 1
        else:
            new_attn_procs[name] = processor
    
    model.set_attn_processor(new_attn_procs)
    
    controller.num_att_layers = cross_att_count
    
    assert cross_att_count != 0, f"No cross-attention layers found in the model. Please check the model structure. Found {cross_att_count} cross-attention layers."


def init_random_noise(device, num_words=500):
    return torch.randn(1, num_words, 1024).to(device)
