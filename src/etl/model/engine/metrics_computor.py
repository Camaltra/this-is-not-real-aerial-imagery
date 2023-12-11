from torch.utils.data import DataLoader
import torch


class BCMetrics:
    @staticmethod
    def compute_accuracy(
        loader: DataLoader,
        model: torch.nn.Module,
        device: str = "mps",
    ) -> float:
        accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                preds = (preds > 0.5).float()
                accuracy += (preds == y).sum() / x.shape[0]
        model.train()
        accuracy /= len(loader)
        return accuracy.item()  # type: ignore

    @staticmethod
    def compute_loss(
        loader: DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        device: str = "mps",
    ) -> float:
        total_loss = 0.0

        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                total_loss += loss_fn(preds, y.float()).item()
        model.train()
        total_loss /= len(loader)
        return total_loss
