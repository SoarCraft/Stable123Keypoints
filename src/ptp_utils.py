import torch.distributions as dist
import numpy as np
import torch
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
        latent = image2latent(ldm, image, device)
        
    noise = torch.randn_like(latent)

    noisy_image = ldm.scheduler.add_noise(
        latent, noise, ldm.scheduler.timesteps[noise_level]
    )
    
    # import ipdb; ipdb.set_trace()

    pred_noise = ldm.unet(noisy_image, 
                          ldm.scheduler.timesteps[noise_level].repeat(noisy_image.shape[0]), 
                          encoder_hidden_states=context.repeat(noisy_image.shape[0], 1, 1))["sample"]
    
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


def register_attention_control(model, controller, feature_upsample_res=256):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            batch_size, sequence_length, dim = hidden_states.shape
            h = self.heads
            q = self.to_q(hidden_states)
            is_cross = encoder_hidden_states is not None

            encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            # sim = torch.matmul(q, k.permute(0, 2, 1)) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim = sim.masked_fill(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = torch.nn.Softmax(dim=-1)(sim)
            attn = attn.clone()
            
            out = torch.matmul(attn, v)

            if (
                is_cross
                and sequence_length <= 32**2
                and len(controller.step_store["attn"]) < 4
            ):
                x_reshaped = hidden_states.reshape(
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
                q = self.head_to_batch_dim(q)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                attn = torch.nn.Softmax(dim=-1)(sim)
                attn = attn.clone()

                attn = controller({"attn": attn}, is_cross, place_in_unet)

            out = self.batch_to_head_dim(out)
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
            is_cross_attention = getattr(net_, 'is_cross_attention', False)
            if is_cross_attention:
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
    
    assert cross_att_count != 0, f"No cross-attention layers found in the model. Please check the model structure. Found {cross_att_count} cross-attention layers."


def init_random_noise(device, num_words=77):
    return torch.randn(1, num_words, 768).to(device)
