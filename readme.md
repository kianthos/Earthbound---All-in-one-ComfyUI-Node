============================================================
EARTHBOUND - ALL-IN-ONE CUSTOM NODE FOR COMFYUI
============================================================

The EarthboundAllInOne node is a comprehensive, single-node 
solution for ComfyUI designed to streamline the high-quality 
image generation process. It integrates base 
generation, latent refinement, neural upscaling, and advanced 
post-processing into a single workflow.

------------------------------------------------------------
1. CORE FEATURES
------------------------------------------------------------

* Integrated Prompting: Positive and negative text boxes are 
  built directly into the node, removing the need for 
  external CLIP Text Encode nodes.

* Three-Pass Pipeline:
    1. Base Pass: Initial image generation.
    2. Refiner Pass: Latent-level refinement to improve 
       details.
    3. Detailer Pass: Post-upscale sampling pass to restore 
       lost high-frequency details.

* Neural Tiled Upscale: High-quality scaling using 
  UPSCALE_MODEL with tiled processing to manage VRAM 
  usage.

* Professional Post-Processing: Includes skin smoothing, 
  sharpening, color correction (contrast, brightness, 
  saturation, gamma), and film grain.

------------------------------------------------------------
2. INPUT PARAMETERS
------------------------------------------------------------

- model/clip/vae: Standard ComfyUI model inputs.
- upscale_model: Model used for the neural upscale pass.
- positive_prompt: Text box for your positive prompt.
- negative_prompt: Text box for your negative prompt.
- resolution: Preset aspect ratios (e.g., 1024x1024).
- upscale_by: Total multiplier for final output size.
- refine_strength: Denoise for the second latent pass.
- upscale_detail_strength: Denoise for the third pass.

------------------------------------------------------------
3. POST-PROCESSING CONTROLS
------------------------------------------------------------

- Smooth Skin: Applies a soft Gaussian blur lerp to reduce 
  skin texture noise.
- Shading Depth: Adjusts the power (gamma-like) of the image 
  to deepen shadows.
- Sharpen: High-pass filter based sharpening.
- Grain: Adds procedural film grain to the final output.
- Color Correction: Independent sliders for Brightness, 
  Contrast, Saturation, and Gamma.

------------------------------------------------------------
4. INSTALLATION
------------------------------------------------------------

1. Navigate to your ComfyUI/custom_nodes/ directory.
2. Create a folder named "EarthboundNodes".
3. Place the node.py file inside that folder.
4. Restart ComfyUI.

------------------------------------------------------------
5. REQUIREMENTS
------------------------------------------------------------

- torch
- torchvision
- Pillow
- numpy