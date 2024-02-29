import os
from datetime import datetime

import torch

from src.dataset import get_train_val_loaders
from src.models import MiniCnn, SmallVGG16Like, VGG16Like
from src.training import inference_aggregator_loop, train_loop

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    print("Run name:", run_name)

    all_folds = False
    num_epochs = 30

    save_directory = os.path.join("saved_models", run_name)
    os.makedirs(save_directory, exist_ok=True)

    if all_folds:
        num_folds = 4
    else:
        num_folds = 1

    for fold in range(num_folds):
        model = MiniCnn().to(device)
        train_loader, val_loader = get_train_val_loaders(
            batch_size=128, num_workers=0, pin_memory=True, fold_id=fold, fold_numbers=4
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = torch.nn.BCEWithLogitsLoss()

        best_val_acc = 0
        for epoch in range(num_epochs):
            train_loss = train_loop(
                model, train_loader, optimizer, loss_function, device
            )
            val_loss, val_acc = inference_aggregator_loop(
                model, val_loader, device, loss_function
            )
            print(
                f"Fold {fold}, Epoch {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_directory, f"fold_{fold}_epoch_{epoch}_acc_{val_acc}.pth"
                    ),
                )

        torch.save(
            model.state_dict(),
            os.path.join(save_directory, f"fold_{fold}_final.pth"),
        )
