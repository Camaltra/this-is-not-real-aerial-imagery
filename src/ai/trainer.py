import torch
from tqdm import tqdm
from torchvision import utils
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
from ai.utils import cycle
from ai.dataset import AerialDataset
from ai.diffusion_process import DiffusionModel


class Trainer:
    def __init__(
        self,
        diffusion_model: DiffusionModel,
        folder: str,
        *,
        train_batch_size: int = 16,
        augment_horizontal_flip: bool = True,
        train_lr: float = 1e-4,
        train_num_steps: int = 100000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        save_and_sample_every: int = 1000,
        num_samples: int = 4,
        results_folder: str = "./results",
        save_best_and_latest_only: bool = False,
    ) -> None:
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        self.step = 0

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.ds = AerialDataset(
            folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip
        )
        self.dl = cycle(
            DataLoader(
                self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True
            )
        )
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self) -> str:
        return "mps"

    def save(self, milestone: int) -> None:
        data = {
            "step": self.step,
            "model": self.model.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "version": "1.0",
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}-final.pt"))

    def load(self, milestone: int, is_old_model: bool = False) -> None:
        data = torch.load(
            str(self.results_folder / f"model-{milestone}-final.pt"),
            map_location=self.device,
        )

        if is_old_model:
            # When coded, the old model has the need of these buffers,
            # but it came to change and, for now don't need anymore
            del data["ema"]["online_model.log_one_minus_alphas_cumprod"]
            del data["ema"]["online_model.sqrt_recip_alphas_cumprod"]
            del data["ema"]["online_model.sqrt_recipm1_alphas_cumprod"]
            del data["ema"]["ema_model.log_one_minus_alphas_cumprod"]
            del data["ema"]["ema_model.sqrt_recip_alphas_cumprod"]
            del data["ema"]["ema_model.sqrt_recipm1_alphas_cumprod"]

        self.model.model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

    def train(self) -> None:
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                data = next(self.dl).to(self.device)
                loss = self.model(data)
                total_loss += loss.item()

                loss.backward()

                pbar.set_description(f"loss: {total_loss:.4f}")

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()

                    with torch.inference_mode():
                        milestone = self.step // self.save_and_sample_every
                        sampled_imgs = self.ema.ema_model.sample(
                            batch_size=self.num_samples
                        )

                    for ix, sampled_img in enumerate(sampled_imgs):
                        utils.save_image(
                            sampled_img,
                            str(
                                self.results_folder
                                / f"sample-{milestone}-{ix}-final.png"
                            ),
                        )

                    self.save(milestone)
                    torch.mps.empty_cache()
                pbar.update(1)
