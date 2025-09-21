# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.distributions as dist
import numpy as np
import torch
from typing import Optional
from tqdm import tqdm
import torch.nn.functional as F
import abc
from src.eval import find_max_pixel, find_k_max_pixels
from src import optimize_token
from PIL import Image
from src.optimize import collect_maps


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
    target/=target.sum(dim=-1, keepdim=True)

    # sort the kl distances between attention_maps_softmax and target
    kl_distances = torch.sum(target * (torch.log(target) - torch.log(attention_maps_softmax)), dim=-1)
    # get the argsort for kl_distances
    kl_distances_argsort = torch.argsort(kl_distances, dim=-1, descending=False)
    
    return torch.tensor(kl_distances_argsort[:top_k]).to(device)


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
        latent = image2latent(ldm, image, device)
        
    noise = torch.randn_like(latent)

    noisy_image = ldm.scheduler.add_noise(
        latent, noise, ldm.scheduler.timesteps[noise_level]
    )
    
    # import ipdb; ipdb.set_trace()

    pred_noise = ldm.unet(noisy_image, 
                          ldm.scheduler.timesteps[noise_level].repeat(noisy_image.shape[0]), 
                          context.repeat(noisy_image.shape[0], 1, 1))["sample"]
    
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
        )
        
        attention_maps.append(_attention_maps)

        controllers[controller].reset()
        
        

    return attention_maps


def image2latent(model, image, device):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            # print the max and min values of the image
            image = torch.from_numpy(image).float() * 2 - 1
            image = image.permute(0, 3, 1, 2).to(device)
            if isinstance(model.vae, torch.nn.DataParallel):
                latents = model.vae.module.encode(image)["latent_dist"].mean
            else:
                latents = model.vae.encode(image)["latent_dist"].mean
            latents = latents * 0.18215
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def latent_step(model, controller, latents, context, t, guidance_scale, low_resource=True):
    if low_resource:
        # noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    # noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    noise_pred = noise_prediction_text
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def register_attention_control_generation(model, controller, target_attn_maps, indices):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward
    
    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
            
    print("cross_att_count")
    print(cross_att_count)

    controller.num_att_layers = cross_att_count



@torch.no_grad()
def text2image_ldm_stable(
    model,
    embedding,
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    target_attn_maps = torch.load("example_attn_maps_indices.pt")
    indices = torch.load("outputs/indices.pt")
    register_attention_control_generation(model, controller, target_attn_maps, indices)
    height = width = 512
    
    
    uncond_input = model.tokenizer(
        [""], padding="max_length", max_length=77, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latents = torch.randn(
        (1, model.unet.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(model.device)
    
    context = [uncond_embeddings, embedding]

    # set timesteps
    # extra_set_kwargs = {"offset": 1}
    extra_set_kwargs = {}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = latent_step(model, controller, latents, context, t, guidance_scale=guidance_scale, low_resource = True)
        controller.reset()
    # latents = latent_step(model, controller, latents, context, t, guidance_scale=guidance_scale, low_resource = True)

    image = latent2image(model.vae, latents)

    return image, latent


def register_attention_control(model, controller, feature_upsample_res=256):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None

            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            # sim = torch.matmul(q, k.permute(0, 2, 1)) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim = sim.masked_fill(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = torch.nn.Softmax(dim=-1)(sim)
            attn = attn.clone()
            
            out = torch.matmul(attn, v)

            if (
                is_cross
                and sequence_length <= 32**2
                and len(controller.step_store["attn"]) < 4
            ):
                x_reshaped = x.reshape(
                    batch_size,
                    int(sequence_length**0.5),
                    int(sequence_length**0.5),
                    dim,
                ).permute(0, 3, 1, 2)
                # upsample to feature_upsample_res**2
                x_reshaped = (
                    F.interpolate(
                        x_reshaped,
                        size=(feature_upsample_res, feature_upsample_res),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(batch_size, -1, dim)
                )

                q = self.to_q(x_reshaped)
                q = self.reshape_heads_to_batch_dim(q)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                attn = torch.nn.Softmax(dim=-1)(sim)
                attn = attn.clone()

                attn = controller({"attn": attn}, is_cross, place_in_unet)

            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")

    controller.num_att_layers = cross_att_count
    
    assert cross_att_count != 0, f"No attention layers found in the model. Please check the model structure. Found {cross_att_count} attention layers."


def init_random_noise(device, num_words=77):
    return torch.randn(1, num_words, 768).to(device)
