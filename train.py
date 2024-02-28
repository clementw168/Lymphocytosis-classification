import torch

from src.dataset import get_train_val_loaders
from src.models import SmallVGG16Like, VGG16Like
from src.training import inference_aggregator_loop, train_loop

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SmallVGG16Like().to(device)

    train_loader, val_loader = get_train_val_loaders(
        batch_size=128, num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.BCEWithLogitsLoss()

    for epoch in range(30):
        train_loss = train_loop(model, train_loader, optimizer, loss_function, device)
        val_loss, val_acc = inference_aggregator_loop(
            model, val_loader, device, loss_function
        )
        print(
            f"Epoch {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc}"
        )
