import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import nodes
import comfy.samplers
import comfy.utils
import comfy.model_management
from PIL import Image
import numpy as np

class EarthboundAllInOne:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "upscale_model": ("UPSCALE_MODEL",),
                # Internal text boxes are now the primary source of conditioning
                "positive_prompt": ("STRING", {"multiline": True, "default": "high quality, masterpiece"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality"}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "auto_grid": (["Disabled", "Enabled"],),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "resolution": (["832x1216", "1216x832", "1024x1024", "1280x720", "720x1280"],),
                
                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "refine_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_detail_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.40, "step": 0.01}),
                
                "smooth_skin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "shading_depth": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "sharpen": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 2.2, "step": 0.01}),
                "grain": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MODEL", "width", "height")
    FUNCTION = "generate_all"
    CATEGORY = "Earthbound"

    def generate_all(self, model, clip, vae, upscale_model, 
                            positive_prompt, negative_prompt,
                            seed, batch_size, auto_grid, steps, cfg, sampler_name, scheduler, denoise,
                            resolution, upscale_by, refine_strength, upscale_detail_strength,
                            smooth_skin, shading_depth, sharpen, brightness, contrast, saturation, gamma, grain):
        
        # --- INTERNAL CLIP ENCODING ---
        tokens_pos = clip.tokenize(positive_prompt)
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
        final_positive = [[cond_pos, {"pooled_output": pooled_pos}]]

        tokens_neg = clip.tokenize(negative_prompt)
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
        final_negative = [[cond_neg, {"pooled_output": pooled_neg}]]
        # -----------------------------

        width, height = map(int, resolution.split('x'))
        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.intermediate_device()

        # PASS 1: Base Generation
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        samples = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, final_positive, final_negative, {"samples":latent}, denoise=denoise)[0]
        
        # PASS 2: Latent Refiner
        if refine_strength > 0:
            samples = nodes.common_ksampler(model, seed + 1, 10, cfg, sampler_name, scheduler, final_positive, final_negative, samples, denoise=refine_strength)[0]
        
        # Decode
        image = vae.decode(samples["samples"])
        del latent, samples
        comfy.model_management.soft_empty_cache()

        # Neural Upscale Logic (TILED)
        upscale_model.to(device)
        img_in = image.movedim(-1, 1).to(device)
        upscaled_image = self.tiled_upscale(img_in, upscale_model, tile_size=512, overlap=32)
        
        upscale_model.to(offload_device)
        del img_in
        comfy.model_management.soft_empty_cache()

        target_w = (int(width * upscale_by) // 8) * 8
        target_h = (int(height * upscale_by) // 8) * 8
        upscaled_image = F.interpolate(upscaled_image, size=(target_h, target_w), mode="bicubic")
        upscaled_image = upscaled_image.movedim(1, -1)

        # PASS 3: Post-Upscale Detailer
        if upscale_detail_strength > 0:
            upscaled_latent = vae.encode(upscaled_image[:,:,:,:3])
            samples = nodes.common_ksampler(model, seed + 2, 6, cfg, sampler_name, scheduler, final_positive, final_negative, {"samples":upscaled_latent}, denoise=upscale_detail_strength)[0]
            
            del upscaled_image, upscaled_latent
            upscaled_image = vae.decode(samples["samples"])
            del samples
            comfy.model_management.soft_empty_cache()

        # Processing Pipeline (CPU)
        img = upscaled_image.permute(0, 3, 1, 2).cpu()
        del upscaled_image

        if smooth_skin > 0:
            blurred_soft = TF.gaussian_blur(img, kernel_size=(9, 9), sigma=(2.0, 2.0))
            img = torch.lerp(img, blurred_soft, smooth_skin * 0.4)
            del blurred_soft
            
        if sharpen > 0:
            blurred_sharp = TF.gaussian_blur(img, kernel_size=(3, 3), sigma=(0.5, 0.5))
            img = img + ((img - blurred_sharp) * sharpen)
            del blurred_sharp
        
        img = img.permute(0, 2, 3, 1)
        img = torch.pow(img.clamp(0, 1), shading_depth)
        img = (img + brightness - 0.5) * contrast + 0.5
        img = torch.pow(img.clamp(1e-5, 1.0), 1.0 / gamma)

        if saturation != 1.0:
            luma = (0.299 * img[:, :, :, 0] + 0.587 * img[:, :, :, 1] + 0.114 * img[:, :, :, 2]).unsqueeze(-1).repeat(1, 1, 1, 3)
            img = torch.lerp(luma, img, saturation)
            del luma
        
        if grain > 0:
            img = img + (torch.randn_like(img) * grain * 0.05 * img)
        
        img_out = img.clamp(0, 1)
        comfy.model_management.soft_empty_cache()

        return (img_out, model, target_w, target_h)

    def tiled_upscale(self, img, model, tile_size, overlap):
        b, c, h, w = img.shape
        output = None
        
        for i in range(0, h, tile_size - overlap):
            for j in range(0, w, tile_size - overlap):
                h_end = min(i + tile_size, h)
                w_end = min(j + tile_size, w)
                tile = img[:, :, i:h_end, j:w_end]
                
                with torch.no_grad():
                    upscaled_tile = model(tile)
                
                if output is None:
                    scale = upscaled_tile.shape[2] // tile.shape[2]
                    output = torch.zeros((b, c, h * scale, w * scale), device=img.device)
                
                output[:, :, i*scale:h_end*scale, j*scale:w_end*scale] = upscaled_tile
        
        return output

NODE_CLASS_MAPPINGS = {"EarthboundAllInOne": EarthboundAllInOne}
NODE_DISPLAY_NAME_MAPPINGS = {"EarthboundAllInOne": "Earthbound - All-in-one"}