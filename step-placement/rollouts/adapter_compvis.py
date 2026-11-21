import torch
import torch.nn as nn
import numpy as np
import sys
import os
from types import SimpleNamespace

# --- CONFIGURATION: Point this to your cloned repo ---
COMPVIS_REPO_PATH = "../latent-diffusion" 
if COMPVIS_REPO_PATH not in sys.path:
    sys.path.append(COMPVIS_REPO_PATH)

try:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
except ImportError:
    print(f"❌ ERROR: Could not import modules from {COMPVIS_REPO_PATH}.")
    print("   Please clone the repo: git clone https://github.com/CompVis/latent-diffusion")
    raise

# --- 1. Fake Output Objects ---
class FakeSchedulerOutput:
    """Mimics diffusers.scheduler.step() return object"""
    def __init__(self, prev_sample, pred_original_sample=None):
        self.prev_sample = prev_sample
        self.pred_original_sample = pred_original_sample

class FakeUNetOutput:
    """Mimics diffusers.unet() return object"""
    def __init__(self, sample):
        self.sample = sample

class CompVisSchedulerAdapter:
    """
    Mimics DDIMScheduler with proper DDIM alpha precomputation.
    """
    def __init__(self, model_wrapper):
        self.wrapper = model_wrapper 
        self.model = model_wrapper.model
        
        self.num_train_timesteps = self.model.num_timesteps
        
        self.config = SimpleNamespace(
            num_train_timesteps=self.model.num_timesteps, 
            prediction_type="epsilon"
        )
        self.timesteps = None
        self._ddim_timesteps = None  # Raw timestep indices
        self._ddim_alphas = None     # Precomputed alphas at DDIM steps
        self._ddim_alphas_prev = None # Precomputed prev alphas
        self._ddim_sqrt_one_minus_alphas = None
        
        self.alphas_cumprod = self.model.alphas_cumprod.to(self.wrapper.device)
        # CRITICAL FIX: Use alphas_cumprod[0] instead of 1.0
        self.final_alpha_cumprod = self.alphas_cumprod[0].clone()

    def set_timesteps(self, num_inference_steps, ddim_eta=0.0):
        """Set up DDIM timesteps and precompute alpha arrays like original DDIM."""
        T = self.model.num_timesteps
        if num_inference_steps > T:
            raise ValueError(f"num_inference_steps={num_inference_steps} exceeds T={T}")

        if num_inference_steps == T:
            # SPECIAL CASE: Full 1000 steps (essentially DDPM, not DDIM subsampling)
            # Use timesteps 0 to T-1 directly, no +1 shift needed
            ddim_timesteps = np.arange(T)
        else:
            # linspace, not `range(0, T, T // n)`: the integer stride silently returns
            # the wrong count whenever n does not divide T (n=400 -> 500 steps).
            # For exact divisors this is identical to the old arange.
            step = T / num_inference_steps
            ddim_timesteps = np.linspace(0, T - step, num_inference_steps).round().astype(int)
            # Add +1 like original DDIM (util.py line 57)
            # "add one to get the final alpha values right"
            ddim_timesteps = ddim_timesteps + 1

        if len(ddim_timesteps) != num_inference_steps:
            raise AssertionError(
                f"requested {num_inference_steps} steps, built {len(ddim_timesteps)}"
            )
        if len(np.unique(ddim_timesteps)) != len(ddim_timesteps):
            raise AssertionError(f"duplicate timesteps for n={num_inference_steps}")

        self._ddim_timesteps = ddim_timesteps
        
        # Reverse for sampling (high to low)
        self.timesteps = torch.from_numpy(ddim_timesteps[::-1].copy()).long().to(self.wrapper.device)
        
        # Precompute alphas at DDIM timesteps (like original make_ddim_sampling_parameters)
        alphas_cumprod_np = self.alphas_cumprod.cpu().numpy()
        alphas = alphas_cumprod_np[ddim_timesteps]
        # For prev, use [alpha[0]] + alphas[:-1] shifted
        alphas_prev = np.asarray([alphas_cumprod_np[0]] + alphas_cumprod_np[ddim_timesteps[:-1]].tolist())
        
        # Compute sigmas for DDIM (eta=0 means deterministic)
        sigmas = ddim_eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
        
        self._ddim_alphas = torch.from_numpy(alphas).float().to(self.wrapper.device)
        self._ddim_alphas_prev = torch.from_numpy(alphas_prev).float().to(self.wrapper.device)
        self._ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - self._ddim_alphas)
        self._ddim_sigmas = torch.from_numpy(sigmas).float().to(self.wrapper.device)
        self._num_inference_steps = num_inference_steps
        
        # Debug info
        # print(f"   📊 DDIM Schedule: {num_inference_steps} steps, eta={ddim_eta}")
        # print(f"   📊 Timesteps (first 5): {self.timesteps[:5].tolist()}")
        # print(f"   📊 Alphas (first 3): {self._ddim_alphas[:3].tolist()}")
        # print(f"   📊 Alphas_prev (first 3): {self._ddim_alphas_prev[:3].tolist()}")
        # print(f"   📊 Final alpha (alphas_cumprod[0]): {self.final_alpha_cumprod.item():.6f}")

    def step(self, model_output, timestep, sample, step_index=None):
        """
        Performs one backward step (t -> t_prev) using precomputed DDIM alphas.
        
        Args:
            model_output: Predicted noise from UNet
            timestep: Current timestep (used to find index if step_index not provided)
            sample: Current noisy sample
            step_index: Optional index into the DDIM schedule (more reliable)
        """
        # Determine step index
        if step_index is not None:
            index = step_index
        else:
            # Find index from timestep
            if isinstance(timestep, (int, np.integer)):
                t_val = timestep
            else:
                t_val = timestep.item() if timestep.dim() == 0 else timestep[0].item()
            
            # Find the index where this timestep appears
            timesteps_list = self.timesteps.cpu().numpy().tolist()
            if t_val in timesteps_list:
                index = timesteps_list.index(t_val)
            else:
                # Fallback: compute approximate index
                index = len(self.timesteps) - 1 - int(t_val * len(self.timesteps) / self.num_train_timesteps)
                index = max(0, min(index, len(self.timesteps) - 1))
        
        # Use precomputed alphas (indexed from the END since timesteps are reversed)
        # The DDIM arrays are in forward order, but we sample in reverse
        ddim_index = len(self.timesteps) - 1 - index
        ddim_index = max(0, min(ddim_index, len(self._ddim_alphas) - 1))
        
        a_t = self._ddim_alphas[ddim_index]
        a_prev = self._ddim_alphas_prev[ddim_index]
        sqrt_one_minus_at = self._ddim_sqrt_one_minus_alphas[ddim_index]
        sigma_t = self._ddim_sigmas[ddim_index]
        
        # Reshape for broadcasting
        b = sample.shape[0]
        a_t = a_t.view(1, 1, 1, 1).expand(b, 1, 1, 1)
        a_prev = a_prev.view(1, 1, 1, 1).expand(b, 1, 1, 1)
        sqrt_one_minus_at = sqrt_one_minus_at.view(1, 1, 1, 1).expand(b, 1, 1, 1)
        sigma_t = sigma_t.view(1, 1, 1, 1).expand(b, 1, 1, 1)
        
        # DDIM update (matching original ddim.py p_sample_ddim)
        # pred_x0 = (x - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        pred_original_sample = (sample - sqrt_one_minus_at * model_output) / a_t.sqrt()
        
        # direction pointing to x_t
        # FIX: Clamp to prevent negative values inside sqrt (can happen with certain eta/scheduler configs)
        dir_xt = (1.0 - a_prev - sigma_t**2).clamp(min=0).sqrt() * model_output
        
        # noise (sigma_t = 0 for DDIM with eta=0)
        noise = sigma_t * torch.randn_like(sample) if sigma_t.sum() > 0 else 0
        
        # x_prev = sqrt(alpha_prev) * pred_x0 + dir_xt + noise
        prev_sample = a_prev.sqrt() * pred_original_sample + dir_xt + noise
        
        return FakeSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

class CompVisUNetAdapter:
    """
    Mimics UNet2DModel. 
    NOT an nn.Module to avoid infinite recursion in .train()/.to() calls.
    """
    def __init__(self, model_wrapper):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        
        # Safely get image size
        unet_img_size = getattr(self.model.model.diffusion_model, 'image_size', 64)
        
        self.config = SimpleNamespace(
            in_channels=self.model.model.diffusion_model.in_channels,
            sample_size=unet_img_size
        )

    # __call__ allows usage like: model.unet(x, t)
    def __call__(self, sample, timestep, encoder_hidden_states=None, **kwargs):
        # 1. Ensure timestep is a tensor on the correct device
        if not torch.is_tensor(timestep):
             timestep = torch.tensor([timestep], device=self.wrapper.device)
        elif timestep.device != self.wrapper.device:
             timestep = timestep.to(self.wrapper.device)
             
        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
            
        # 2. Handle Conditioning
        if encoder_hidden_states is not None:
            # Explicit context passed from outside
            c = encoder_hidden_states
            
        elif getattr(self.model, 'cond_stage_model', None) is None:
            # --- CRITICAL FIX FOR CRASH ---
            # Unconditional Model (LSUN Church/Bedroom). 
            # It has no text/class encoder. Pass None.
            c = None
            
        else:
            # --- CASE: Conditional Model (ImageNet / Text) ---
            batch_size = sample.shape[0]
            cond_key = getattr(self.model, 'cond_stage_key', 'image')
            is_class_model = (cond_key == 'class_label') or hasattr(self.model.cond_stage_model, 'key')

            if is_class_model:
                key = getattr(self.model.cond_stage_model, 'key', cond_key)
                
                if 'class_labels' in kwargs and kwargs['class_labels'] is not None:
                    labels = kwargs['class_labels']
                else:
                    # Fallback to Class 0 only if nothing is requested
                    labels = torch.zeros(batch_size, device=self.wrapper.device, dtype=torch.long)
                
                input_c = {key: labels}
            else:
                # Text-to-Image: Pass empty string
                input_c = [""] * batch_size

            # Encode the dummy input
            c = self.model.get_learned_conditioning(input_c)

        # 3. Call internal CompVis apply_model
        noise_pred = self.model.apply_model(sample, timestep, c)
        
        return FakeUNetOutput(sample=noise_pred)
        
# --- 4. Main Model Adapter ---
class CompVisAdapter(nn.Module):
    def __init__(self, config_path, ckpt_path, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        
        print(f"🔌 Loading CompVis Adapter from: {ckpt_path}")
        
        # ... (Loading code remains the same) ...
        config = OmegaConf.load(config_path)
        
        if "first_stage_config" in config.model.params:
            if "ckpt_path" in config.model.params.first_stage_config.params:
                print("⚠️  Removing hardcoded 'ckpt_path' from config to prevent FileNotFoundError...")
                del config.model.params.first_stage_config.params["ckpt_path"]
                
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        self.unet = CompVisUNetAdapter(self)
        self.scheduler = CompVisSchedulerAdapter(self)
        
        # --- ROBUST CHANNEL DETECTION ---
        # 1. Try UNet Input Channels (Most Reliable)
        if hasattr(self.model.model, 'diffusion_model') and hasattr(self.model.model.diffusion_model, 'in_channels'):
            self.latent_channels = self.model.model.diffusion_model.in_channels
        # 2. Try VAE 'embed_dim' (Common in CompVis)
        elif hasattr(self.model.first_stage_model, 'embed_dim'):
            self.latent_channels = self.model.first_stage_model.embed_dim
        # 3. Fallback
        else:
            print("⚠️ Warning: Could not detect latent channels. Defaulting to 4.")
            self.latent_channels = 4

        # --- LATENT SIZE DETECTION ---
        if hasattr(self.model.model.diffusion_model, 'image_size'):
            self.latent_size = self.model.model.diffusion_model.image_size
        else:
            self.latent_size = 32 # Default fallback

        self.img_size = 256 # Default assumption for LSUN/CelebA
        
        # Expose num_timesteps for compatibility with rl_trainer.py
        self.num_timesteps = self.model.num_timesteps
        
        print(f"✅ Adapter Configured: {self.latent_channels} channels, {self.latent_size}x{self.latent_size} latents, {self.num_timesteps} timesteps")

    def to(self, device):
        # Handle device movement
        device = torch.device(device)
        if str(self.device) != str(device):
            self.device = device
            self.model.to(device)
            # Update internal caches
            if hasattr(self.scheduler, 'alphas_cumprod'):
                self.scheduler.alphas_cumprod = self.model.alphas_cumprod.to(device)
                self.scheduler.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
                # Also move precomputed DDIM arrays if they exist
                if self.scheduler._ddim_alphas is not None:
                    self.scheduler._ddim_alphas = self.scheduler._ddim_alphas.to(device)
                    self.scheduler._ddim_alphas_prev = self.scheduler._ddim_alphas_prev.to(device)
                    self.scheduler._ddim_sqrt_one_minus_alphas = self.scheduler._ddim_sqrt_one_minus_alphas.to(device)
                    self.scheduler._ddim_sigmas = self.scheduler._ddim_sigmas.to(device)
        return self

    @torch.no_grad()
    def _encode_image_to_latents(self, x_0):
        encoder_posterior = self.model.encode_first_stage(x_0)
        z = self.model.get_first_stage_encoding(encoder_posterior)
        return z

    @torch.no_grad()
    def _decode_latents_to_image(self, latents):
        x_rec = self.model.decode_first_stage(latents)
        return x_rec

    # def _decode_latents_to_image(self, latents):
    #     # Scale UP: Return to VAE's native high-variance space
    #     # Magic number for CompVis/Stable Diffusion models
    #     # print(f"DEBUG: Latent Mean: {latents.mean():.2f}, Std: {latents.std():.2f}, Min: {latents.min():.2f}, Max: {latents.max():.2f}")
    #     # latents = (1.0 / 0.18215) * latents
    #     x_rec = self.model.decode_first_stage(latents)
    #     return x_rec.clamp(-1, 1)

    # @torch.no_grad()
    # def _denoise_step_latents(self, z_t, t_start_batch, t_end_batch, eta=0.0):
    #     """
    #     Performs the jump from t_start -> t_end in latent space.
    #     """
    #     c = None
    #     pred_epsilon = self.model.apply_model(z_t, t_start_batch, c)

    #     alphas = self.scheduler.alphas_cumprod
        
    #     alpha_start = alphas.gather(0, t_start_batch).view(-1, 1, 1, 1)
        
    #     # Handle t_end < 0 for the clean step
    #     alpha_end = torch.where(
    #         t_end_batch.view(-1, 1, 1, 1) >= 0,
    #         alphas.gather(0, t_end_batch.clamp(min=0)).view(-1, 1, 1, 1),
    #         torch.tensor(1.0, device=self.device)
    #     )

    #     beta_start_sqrt = (1.0 - alpha_start).sqrt()
    #     pred_z0 = (z_t - beta_start_sqrt * pred_epsilon) / alpha_start.sqrt().clamp(min=1e-6)
        
    #     sigma = 0.0 
    #     noise = 0.0
        
    #     pred_dir_zt = (1.0 - alpha_end - sigma**2).clamp(min=0).sqrt() * pred_epsilon
    #     z_end = alpha_end.sqrt() * pred_z0 + pred_dir_zt + sigma * noise
        
    #     return z_end, pred_epsilon

    @torch.no_grad()
    def _denoise_step_latents(self, z_t, t_start_batch, t_end_batch, eta=0.0, class_labels=None):
        """
        Performs the jump from t_start -> t_end in latent space.
        Updated to accept class_labels for conditional generation.
        
        CRITICAL FIX: Use alphas_cumprod[0] instead of 1.0 for final step.
        """
        # Get noise prediction from UNet
        pred_epsilon = self.unet(z_t, t_start_batch, class_labels=class_labels).sample

        alphas = self.scheduler.alphas_cumprod
        
        # Get alpha at start timestep
        alpha_start = alphas.gather(0, t_start_batch.clamp(min=0)).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_start = (1.0 - alpha_start).sqrt()
        
        # CRITICAL FIX: Use alphas_cumprod[0] instead of 1.0 for final step
        # This matches the original DDIM behavior
        final_alpha = self.scheduler.final_alpha_cumprod  # This is now alphas_cumprod[0]
        
        # Handle t_end < 0 for the final clean step
        alpha_end = torch.where(
            t_end_batch.view(-1, 1, 1, 1) >= 0,
            alphas.gather(0, t_end_batch.clamp(min=0)).view(-1, 1, 1, 1),
            final_alpha.view(1, 1, 1, 1).expand(z_t.shape[0], 1, 1, 1)
        )

        # Compute predicted x_0 (matching DDIM formula)
        pred_z0 = (z_t - sqrt_one_minus_alpha_start * pred_epsilon) / alpha_start.sqrt().clamp(min=1e-8)
        
        # Direction pointing to x_t (DDIM with eta=0, no noise)
        sqrt_one_minus_alpha_end = (1.0 - alpha_end).clamp(min=0).sqrt()
        pred_dir_zt = sqrt_one_minus_alpha_end * pred_epsilon
        
        # Compute x_prev
        z_end = alpha_end.sqrt() * pred_z0 + pred_dir_zt
        
        return z_end, pred_epsilon, pred_z0

def verify_ddim_against_original(adapter, num_samples=1, seed=42):
    """
    Verify our adapter's DDIM sampling against the original DDIMSampler.
    This helps diagnose quality issues by comparing outputs.
    
    Memory-efficient version: runs original first, clears memory, then adapter.
    """
    import sys
    import gc
    sys.path.insert(0, COMPVIS_REPO_PATH)
    from ldm.models.diffusion.ddim import DDIMSampler
    
    device = adapter.device
    model = adapter.model
    num_steps = len(adapter.scheduler.timesteps)
    
    print(f"\n🔍 Verifying DDIM against original sampler ({num_steps} steps, {num_samples} sample(s))...")
    
    # Set seed and create starting noise
    torch.manual_seed(seed)
    z_start = torch.randn(num_samples, 3, 64, 64, device=device)
    z_start_cpu = z_start.cpu().clone()  # Save for later
    
    # --- Run original DDIM sampler ---
    original_sampler = DDIMSampler(model)
    original_sampler.make_schedule(ddim_num_steps=num_steps, ddim_eta=0.0, verbose=False)
    
    with torch.no_grad():
        samples_original, _ = original_sampler.sample(
            S=num_steps,
            batch_size=num_samples,
            shape=[3, 64, 64],
            conditioning=None,
            verbose=False,
            x_T=z_start.clone(),
            eta=0.0
        )
    
    # Save original results and alpha values
    samples_original_cpu = samples_original.cpu().clone()
    orig_alphas = original_sampler.ddim_alphas.cpu().clone()  # It's a tensor, not numpy
    
    # Clean up original sampler to free memory
    del original_sampler, samples_original, z_start
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- Run adapter sampling ---
    z_start = z_start_cpu.to(device)
    samples_adapter = z_start.clone()
    timesteps = adapter.scheduler.timesteps
    
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            t_batch = torch.full((num_samples,), t.item(), device=device, dtype=torch.long)
            noise_pred = adapter.unet(samples_adapter, t_batch).sample
            output = adapter.scheduler.step(noise_pred, t.item(), samples_adapter, step_index=step_idx)
            samples_adapter = output.prev_sample
    
    samples_adapter_cpu = samples_adapter.cpu()
    
    # Clean up
    del z_start, samples_adapter
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- Compare results on CPU ---
    diff = (samples_original_cpu - samples_adapter_cpu).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"   📊 Original latent stats: mean={samples_original_cpu.mean():.4f}, std={samples_original_cpu.std():.4f}")
    print(f"   📊 Adapter latent stats:  mean={samples_adapter_cpu.mean():.4f}, std={samples_adapter_cpu.std():.4f}")
    print(f"   📊 Difference: max={max_diff:.6f}, mean={mean_diff:.6f}")
    
    if max_diff < 0.01:
        print("   ✅ Adapter matches original DDIM!")
    elif max_diff < 0.1:
        print("   ⚠️  Small differences detected - may affect quality slightly")
    else:
        print("   ❌ Significant differences - investigate further!")
        
        # Debug: print alpha comparison
        print("\n   Debug - Alpha comparison at key steps:")
        for i in [0, num_steps//2, num_steps-1]:
            orig_alpha = orig_alphas[num_steps - 1 - i].item()
            adapter_alpha = adapter.scheduler._ddim_alphas[num_steps - 1 - i].item()
            print(f"      Step {i}: orig_alpha={orig_alpha:.6f}, adapter_alpha={adapter_alpha:.6f}")
    
    # Decode on GPU one at a time to save memory
    with torch.no_grad():
        img_original = model.decode_first_stage(samples_original_cpu.to(device)).cpu()
        torch.cuda.empty_cache()
        img_adapter = model.decode_first_stage(samples_adapter_cpu.to(device)).cpu()
        torch.cuda.empty_cache()
    
    img_diff = (img_original - img_adapter).abs().mean().item()
    print(f"   📊 Decoded image difference: {img_diff:.6f}")
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'img_diff': img_diff
    }


def create_compvis_adapter(config_path, ckpt_path, device='cuda', n_timesteps=50, verify=False):
    if n_timesteps is None:
        n_timesteps = 50
        print(f"⚠️  n_timesteps was None, defaulting to {n_timesteps} steps for CompVis Adapter.")
        
    adapter = CompVisAdapter(config_path, ckpt_path, device)
    adapter.scheduler.set_timesteps(n_timesteps)
    
    # Optionally verify against original DDIM
    if verify:
        verify_ddim_against_original(adapter)
    
    return adapter