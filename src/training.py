import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()

    train_loss = 0
    for images, tabular, labels, ids in tqdm(dataloader):
        images, tabular, labels = (
            images.to(device),
            tabular.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()
        predictions = model(images, tabular)
        loss = loss_function(predictions, labels)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)


def inference_aggregator_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_function: torch.nn.Module,
) -> tuple[float, float]:
    model.eval()

    val_loss = 0
    labels_acc = []
    ids_acc = []
    pred_acc = []

    with torch.no_grad():
        for images, tabular, labels, ids in tqdm(dataloader):
            images, tabular, labels = (
                images.to(device),
                tabular.to(device),
                labels.to(device),
            )

            predictions = model(images, tabular)

            val_loss += loss_function(predictions, labels).item()

            labels_acc.append(labels.cpu().numpy())
            ids_acc.append(ids.numpy())
            pred_acc.append(predictions.cpu().numpy())

    labels_acc = np.concatenate(labels_acc)
    ids_acc = np.concatenate(ids_acc)
    pred_acc = np.concatenate(pred_acc)

    unique_ids = np.unique(ids_acc)
    patient_labels = np.zeros(len(unique_ids))
    aggregated_predictions = np.zeros(len(unique_ids))
    for patient_index, unique_id in enumerate(unique_ids):
        mask = ids_acc == unique_id
        patient_labels[patient_index] = int(labels_acc[mask].mean() > 0.5)
        aggregated_predictions[patient_index] = int(pred_acc[mask].mean() > 0.5)

    return val_loss / len(dataloader), balanced_accuracy_score(
        patient_labels, aggregated_predictions
    )
