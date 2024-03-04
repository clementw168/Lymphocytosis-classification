import os
from datetime import datetime

import torch

from src.dataset import get_train_val_loaders
from src.models import MODEL_DICT
from src.training import get_best_threshold, inference_aggregator_loop, train_loop

if __name__ == "__main__":
    all_folds = True
    save = True
    patient_wise_split = False

    model_name = "MobileNetV2Tab"
    num_epochs = 20

    run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_name}"
    print("Run name:", run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save:
        save_directory = os.path.join("saved_models", run_name)
        os.makedirs(save_directory, exist_ok=True)

    if all_folds:
        num_folds = 4
    else:
        num_folds = 1

    for fold in range(num_folds):
        model = MODEL_DICT[model_name]().to(device)
        train_loader, val_loader = get_train_val_loaders(
            batch_size=64,
            num_workers=0,
            pin_memory=True,
            fold_id=fold,
            fold_numbers=4,
            patient_wise=patient_wise_split,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_function = torch.nn.BCEWithLogitsLoss()

        best_val_acc = 0
        for epoch in range(num_epochs):
            train_loss = train_loop(
                model, train_loader, optimizer, loss_function, device
            )
            patient_labels, aggregated_predictions, unique_ids, val_loss = (
                inference_aggregator_loop(model, val_loader, device, loss_function)
            )

            _, _, val_acc = get_best_threshold(patient_labels, aggregated_predictions)

            print(
                f"Fold {fold}, Epoch {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            save_directory,
                            f"fold_{fold}_epoch_{epoch}_acc_{val_acc}.pth",
                        ),
                    )

        if save:
            torch.save(
                model.state_dict(),
                os.path.join(save_directory, f"fold_{fold}_final.pth"),
            )
