from ai.schedulers import (
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
)
from ai.utils import (
    identity,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    extract,
)
import torch
import torch.nn as nn
from torch.functional import F
from tqdm import tqdm


class DiffusionModel(nn.Module):
    SCHEDULER_MAPPING = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        *,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        schedule_fn_kwargs: dict | None = None,
        auto_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model = model

        self.channels = self.model.channels
        self.image_size = image_size

        self.beta_schedule_fn = self.SCHEDULER_MAPPING.get(beta_schedule)
        if self.beta_schedule_fn is None:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}

        betas = self.beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        timesteps, *_ = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = timesteps

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @torch.inference_mode()
    def p_sample(self, x: torch.Tensor, timestamp: int) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        batched_timestamps = torch.full(
            (b,), timestamp, device=device, dtype=torch.long
        )

        preds = self.model(x, batched_timestamps)

        betas_t = extract(self.betas, batched_timestamps, x.shape)
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas, batched_timestamps, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, batched_timestamps, x.shape
        )

        predicted_mean = sqrt_recip_alphas_t * (
            x - betas_t * preds / sqrt_one_minus_alphas_cumprod_t
        )

        if timestamp == 0:
            return predicted_mean
        else:
            posterior_variance = extract(
                self.posterior_variance, batched_timestamps, x.shape
            )
            noise = torch.randn_like(x)
            return predicted_mean + torch.sqrt(posterior_variance) * noise

    @torch.inference_mode()
    def p_sample_loop(
        self, shape: tuple, return_all_timesteps: bool = False
    ) -> torch.Tensor:
        batch, device = shape[0], "mps"

        img = torch.randn(shape, device=device)
        # This cause me a RunTimeError on MPS device due to MPS back out of memory
        # No ideas how to resolve it at this point

        # imgs = [img]

        for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            img = self.p_sample(img, t)
            # imgs.append(img)

        ret = img  # if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    def sample(
        self, batch_size: int = 16, return_all_timesteps: bool = False
    ) -> torch.Tensor:
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        return self.p_sample_loop(shape, return_all_timesteps=return_all_timesteps)

    def q_sample(
        self, x_start: torch.Tensor, t: int, noise: torch.Tensor = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_loss(
        self,
        x_start: torch.Tensor,
        t: int,
        noise: torch.Tensor = None,
        loss_type: str = "l2",
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noised = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.model(x_noised, t)

        if loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        else:
            raise ValueError(f"unknown loss type {loss_type}")
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, device, img_size = *x.shape, x.device, self.image_size
        assert h == w == img_size, f"image size must be {img_size}"

        timestamp = torch.randint(0, self.num_timesteps, (1,)).long().to(device)
        x = self.normalize(x)
        return self.p_loss(x, timestamp)
