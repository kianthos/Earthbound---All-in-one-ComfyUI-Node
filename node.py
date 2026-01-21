import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import nodes
import comfy.samplers
import comfy.utils
import comfy.model_management as mm
import folder_paths 
import numpy as np
import random
import json
import os
import gc

# --- ROBUST DEPENDENCY HANDLING ---
upscale_nodes = None
try:
    import comfy_extras.nodes_upscale_model as upscale_nodes_module
    upscale_nodes = upscale_nodes_module
except ImportError:
    print("Earthbound Ultimate Notice: 'comfy_extras' not found. Upscale model support disabled.")

class EarthboundUltimate:
    def __init__(self):
        try:
            self.recipe_path = os.path.join(folder_paths.get_input_directory(), "earthbound_ultimate_recipes.json")
            if not os.path.exists(self.recipe_path):
                with open(self.recipe_path, "w") as f:
                    json.dump({}, f)
        except Exception:
            self.recipe_path = os.path.join(os.path.dirname(__file__), "recipes.json")
            if not os.path.exists(self.recipe_path):
                with open(self.recipe_path, "w") as f:
                    json.dump({}, f)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # --- CKPT MERGER ---
                "ckpt 1": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt 1 strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ckpt 2": (["None"] + folder_paths.get_filename_list("checkpoints"), ),
                "ckpt 2 strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ckpt 3": (["None"] + folder_paths.get_filename_list("checkpoints"), ),
                "ckpt 3 strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ckpt 4": (["None"] + folder_paths.get_filename_list("checkpoints"), ),
                "ckpt 4 strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ckpt 5": (["None"] + folder_paths.get_filename_list("checkpoints"), ),
                "ckpt 5 strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # --- LORA STACK ---
                "lora 1 name": (["None"] + folder_paths.get_filename_list("loras"), ),
                "lora 1 strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora 2 name": (["None"] + folder_paths.get_filename_list("loras"), ),
                "lora 2 strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora 3 name": (["None"] + folder_paths.get_filename_list("loras"), ),
                "lora 3 strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                
                "internal vae": (["None"] + folder_paths.get_filename_list("vae"), ),

                # --- GENERATION SETTINGS ---
                "positive prompt": ("STRING", {"multiline": True, "default": "masterpiece, high quality"}),
                "negative prompt": ("STRING", {"multiline": True, "default": "lowres, blurry, bad anatomy"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed mode": (["Fixed", "Randomize"],),
                "internal batch size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "res mode": (["Preset", "Custom"],),
                "preset res": (["1024x1024", "832x1216", "1216x832", "1280x720", "720x1280"],),
                "custom width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "custom height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 7.0}),
                "sampler name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0}),

                # --- LATENT CONTROLS ---
                "latent_upscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),

                # --- POST-PROCESSING ---
                "style morph": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "texture pop": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "clarity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sharpen": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "edge definition": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "smooth skin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chromatic aberration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 1}),
                "shading depth": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 2.2, "step": 0.01}),
                "grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # --- SYSTEM ---
                "auto precision": (["Disabled", "Enabled"], {"default": "Disabled"}),
                "recipe action": (["Apply", "Save current as Slot"],),
                "recipe slot": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
            "optional": {
                "model in": ("MODEL",), "clip in": ("CLIP",), "vae in": ("VAE",),
                "upscale model name": (["None"] + folder_paths.get_filename_list("upscale_models"), ),
                "batch size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "control_net": ("CONTROL_NET",), "control_img": ("IMAGE",),
                "mask": ("MASK",), "latent_in": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE")
    FUNCTION = "generate_all"
    CATEGORY = "Earthbound"

    def generate_all(self, **kwargs):
        # --- RECIPE PERSISTENCE ---
        action, slot = kwargs.get("recipe action", "Apply"), str(kwargs.get("recipe slot", 1))
        recipes = {}
        if os.path.exists(self.recipe_path):
            with open(self.recipe_path, "r") as f: recipes = json.load(f)
        if action == "Save current as Slot":
            save_data = {k: v for k, v in kwargs.items() if not isinstance(v, (torch.Tensor, object)) and k not in ["recipe action", "recipe slot"]}
            recipes[slot] = save_data
            with open(self.recipe_path, "w") as f: json.dump(recipes, f, indent=4)
        elif action == "Apply" and slot in recipes:
            kwargs.update(recipes[slot])

        device = mm.get_torch_device()
        def purge():
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            mm.soft_empty_cache()

        # 1. MODEL MERGE PHASE
        if kwargs.get('model in') is not None:
            model, clip, vae = kwargs['model in'], kwargs['clip in'], kwargs['vae in']
        else:
            model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(kwargs['ckpt 1'])
        
        if kwargs.get('internal vae') not in [None, "None"]:
            vae = nodes.VAELoader().load_vae(kwargs['internal vae'])[0]

        # Normalized Checkpoint Merge
        strengths = [kwargs['ckpt 1 strength'], kwargs['ckpt 2 strength'], kwargs['ckpt 3 strength'], kwargs['ckpt 4 strength'], kwargs['ckpt 5 strength']]
        ckpts = [kwargs['ckpt 1'], kwargs['ckpt 2'], kwargs['ckpt 3'], kwargs['ckpt 4'], kwargs['ckpt 5']]
        total_s = sum(strengths) if sum(strengths) > 0 else 1.0
        for i in range(1, 5):
            if ckpts[i] != "None" and strengths[i] > 0:
                s = strengths[i] / total_s
                m_out = nodes.CheckpointLoaderSimple().load_checkpoint(ckpts[i])
                model.add_patches(m_out[0].get_key_patches(), s, 1.0 - s)
                del m_out; purge()

        # LoRA Integration
        loras = [(kwargs['lora 1 name'], kwargs['lora 1 strength']), (kwargs['lora 2 name'], kwargs['lora 2 strength']), (kwargs['lora 3 name'], kwargs['lora 3 strength'])]
        for name, strnt in loras:
            if name != "None" and strnt != 0:
                l_data = comfy.utils.load_torch_file(folder_paths.get_full_path("loras", name), safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(model, clip, l_data, strnt, strnt)
                del l_data; purge()

        # 2. SAMPLING & LATENT OPS
        p_toks, n_toks = clip.tokenize(kwargs['positive prompt']), clip.tokenize(kwargs['negative prompt'])
        cond_p, pool_p = clip.encode_from_tokens(p_toks, return_pooled=True)
        cond_n, pool_n = clip.encode_from_tokens(n_toks, return_pooled=True)
        pos, neg = [[cond_p, {"pooled_output": pool_p}]], [[cond_n, {"pooled_output": pool_n}]]
        if kwargs.get("control_net") and kwargs.get("control_img") is not None:
            pos = nodes.ControlNetApply().apply_controlnet(pos, kwargs["control_net"], kwargs["control_img"], 1.0)[0]

        w, h = (map(int, kwargs['preset res'].split('x')) if kwargs['res mode'] == "Preset" else (kwargs['custom width'], kwargs['custom height']))
        batch = kwargs.get('batch size', kwargs.get('internal batch size', 1))
        latent = kwargs.get("latent_in", {"samples": torch.zeros([batch, 4, h // 8, w // 8])})
        if kwargs.get("mask") is not None:
            latent = latent.copy(); latent["noise_mask"] = kwargs["mask"]
        
        seed = random.randint(0, 0xffffffffffffffff) if kwargs['seed mode'] == "Randomize" else kwargs['seed']
        samples = nodes.common_ksampler(model, seed, kwargs['steps'], kwargs['cfg'], kwargs['sampler name'], kwargs['scheduler'], pos, neg, latent, denoise=kwargs['denoise'])[0]
        if kwargs["latent_upscale"] > 1.0:
            samples = nodes.LatentUpscaleBy().upscale(samples, kwargs["upscale_method"], kwargs["latent_upscale"])[0]

        # 3. DECODE & POST-PROCESSING
        purge()
        image = vae.decode(samples["samples"])
        
        upscale_name = kwargs.get('upscale model name', "None")
        if upscale_nodes is not None and upscale_name != "None":
            upscale_loader = upscale_nodes.UpscaleModelLoader()
            upscale_model = upscale_loader.load_model(upscale_name)[0]
            upscale_model.to(device)
            img_in = image.movedim(-1, 1).to(device)
            upscaled = upscale_model(img_in).movedim(1, -1)
            upscale_model.to("cpu"); del upscale_model; purge()
        else:
            upscaled = image.to(device)

        # 4. FINAL EFFECT PIPELINE
        img_t = upscaled.permute(0, 3, 1, 2)
        if kwargs.get("auto precision") == "Enabled": img_t = img_t.half()

        ca = int(kwargs['chromatic aberration'])
        if ca > 0:
            r = torch.roll(img_t[:, 0:1, :, :], shifts=(ca, ca), dims=(2, 3))
            b = torch.roll(img_t[:, 2:3, :, :], shifts=(-ca, -ca), dims=(2, 3))
            img_t = torch.cat([r, img_t[:, 1:2, :, :], b], dim=1)

        b3, b5 = TF.gaussian_blur(img_t, [3, 3], [0.5, 0.5]), TF.gaussian_blur(img_t, [5, 5], [1.0, 1.0])
        morph = kwargs['style morph']
        if morph < 0:
            img_t = torch.lerp(img_t, b5, abs(morph) * 0.6)
            img_t += ((img_t - b3) * abs(morph) * 1.2)
        elif morph > 0:
            img_t += ((img_t - b5) * morph * 0.8)

        if kwargs['sharpen'] > 0: img_t += (img_t - b3) * kwargs['sharpen']
        if kwargs['texture pop'] > 0: img_t += ((img_t - b5) * kwargs['texture pop'])
        if kwargs['edge definition'] > 0: img_t += ((img_t - b3) * kwargs['edge definition'])
        if kwargs['clarity'] > 0: 
            img_t = torch.lerp(img_t, img_t + (img_t - TF.gaussian_blur(img_t, [21, 21], [5.0, 5.0])), kwargs['clarity'] * 0.5)
        if kwargs['smooth skin'] > 0: 
            img_t = torch.lerp(img_t, TF.gaussian_blur(img_t, [9, 9], [2.0, 2.0]), kwargs['smooth skin'] * 0.4)
            
        img = img_t.permute(0, 2, 3, 1)
        img = torch.pow(img.clamp(1e-5, 1.0), 1.0 / kwargs['gamma'])
        img = torch.pow(img.clamp(0, 1), kwargs['shading depth'] * (1.0 + (morph * 0.2 if morph > 0 else 0)))
        img = (img + kwargs['brightness'] - 0.5) * kwargs['contrast'] + 0.5
        if kwargs['saturation'] != 1.0:
            luma = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).unsqueeze(-1)
            img = torch.lerp(luma, img, kwargs['saturation'])
        if kwargs['grain'] > 0: img += (torch.randn_like(img) * kwargs['grain'] * 0.05 * img)

        return (img.clamp(0, 1).cpu().float(), model, clip, vae)

NODE_CLASS_MAPPINGS = {"EarthboundUltimate": EarthboundUltimate}
NODE_DISPLAY_NAME_MAPPINGS = {"EarthboundUltimate": "Earthbound Ultimate"}