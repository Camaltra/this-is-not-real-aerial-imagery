from ai.models.u_net import UNet
import torch
from ai.diffusion_process import DiffusionModel
from ai.trainer import Trainer

if __name__ == "__main__":
    torch.mps.empty_cache()
    model = UNet(64, channels=3).to("mps")
    diffusion_model = DiffusionModel(model, image_size=128)
    trainer = Trainer(diffusion_model, "../data/training")
    trainer.load(37, is_old_model=True)
    trainer.train()
