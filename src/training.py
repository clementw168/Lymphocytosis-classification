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
    loss_function: torch.nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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

            if loss_function:
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
        aggregated_predictions[patient_index] = pred_acc[mask].mean()

    return (
        patient_labels,
        aggregated_predictions,
        unique_ids,
        val_loss / len(dataloader),
    )


def get_best_threshold(
    patient_labels: np.ndarray,
    aggregated_predictions: np.ndarray,
    print_all: bool = False,
) -> tuple[float, float, float]:
    thresholds_sigmoid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    thresholds_logits = np.log(thresholds_sigmoid / (1 - thresholds_sigmoid))

    best_threshold_index = []
    best_balanced_accuracy = 0

    for i, threshold in enumerate(thresholds_logits):
        balanced_accuracy = balanced_accuracy_score(
            patient_labels, aggregated_predictions > threshold
        )

        if print_all:
            print(
                f"Threshold: {thresholds_sigmoid[i]}, Balanced accuracy: {balanced_accuracy}"
            )

        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_threshold_index = [i]

        elif balanced_accuracy == best_balanced_accuracy:
            best_threshold_index.append(i)

    best_threshold_index = best_threshold_index[len(best_threshold_index) // 2]

    return (
        thresholds_sigmoid[best_threshold_index],
        thresholds_logits[best_threshold_index],
        best_balanced_accuracy,
    )
