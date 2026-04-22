import torch
from datasets import MyDataset
import numpy as np

def flip_labels(ys, alpha):
    # Create copy of original data
    new_ys = np.copy(ys)

    # Calculate number of samples to flip
    n_samples = len(ys)
    n_flip = int(n_samples * alpha)

    # Ensure n_flip is even for balanced allocation
    if n_flip % 2 == 1:
        n_flip -= 1

    # Split dataset into first and second halves
    first_half = np.arange(n_samples // 2)
    second_half = np.arange(n_samples // 2, n_samples)

    # Randomly select half of flip samples from each half
    flip_indices_first = np.random.choice(first_half, n_flip // 2, replace=False)
    flip_indices_second = np.random.choice(second_half, n_flip // 2, replace=False)

    # Combine flip indices from both halves
    flip_indices = np.concatenate([flip_indices_first, flip_indices_second])

    # Flip selected labels (0->1, 1->0)
    new_ys[flip_indices] = 1 - new_ys[flip_indices]

    return new_ys, flip_indices


def data_poison(xs, alpha, xes=1, ordered=True):
    """Poison the dataset by selecting half of the samples from each half of the dataset."""
    # Create copy of original data
    poisoned_xs = np.copy(xs)

    # Calculate number of samples to poison
    n_samples = len(xs)
    n_poison = int(n_samples * alpha)

    # Ensure n_poison is even for balanced allocation
    if n_poison % 2 != 0:
        n_poison += 1

    # Split dataset into first and second halves
    mid_point = n_samples // 2
    first_half_indices = np.arange(mid_point)
    second_half_indices = np.arange(mid_point, n_samples)

    # Select half of poison samples from each half
    n_poison_per_half = n_poison // 2
    first_half_poison = np.random.choice(
        first_half_indices, n_poison_per_half, replace=False
    )
    second_half_poison = np.random.choice(
        second_half_indices, n_poison_per_half, replace=False
    )

    # Combine selected indices
    poison_indices = np.concatenate([first_half_poison, second_half_poison])

    # Replace selected samples
    if ordered:
        for idx in poison_indices:
            poisoned_xs[idx] = xes[idx]
    else:
        # Replace with random complex values
        thetas = np.random.normal(0, 1e-1, size=[len(poison_indices), 2, len(xs[0])])
        for i, idx in enumerate(poison_indices):
            if xs.dtype == "complex128":
                poisoned_xs[idx] = thetas[i, 0] + 1.0j * thetas[i, 1]
            else:
                poisoned_xs[idx] = thetas[i, 0]
            poisoned_xs[idx] = poisoned_xs[idx] / np.linalg.norm(poisoned_xs[idx])

    return poisoned_xs, poison_indices


def poison(dataloader, poison_x=0, poison_y=0):
    """
    Args:
        dataloader: The input dataloader to be poisoned.
        poison_x: alpha for data_poison (feature poisoning ratio)
        poison_y: alpha for flip_labels (label flipping ratio)
    Returns:
        poisonloader: A new DataLoader with poisoned data.
    """
    data_list = []
    labels_list = []

    for feed_dict in dataloader:
        data_list.append(feed_dict['image'])
        labels_list.append(feed_dict['digit'])

    data = torch.cat(data_list).numpy()
    labels = torch.cat(labels_list).numpy()

    # Ensure equal number of label 0 and label 1 samples
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    n_samples = min(len(idx0), len(idx1))

    idx0 = idx0[:n_samples]
    idx1 = idx1[:n_samples]

    # Sort data by labels: first half label 0, second half label 1
    sort_idx = np.concatenate([idx0, idx1])
    data = data[sort_idx]
    labels = labels[sort_idx]

    # Apply poisoning
    if poison_y > 0.001:
        labels, _ = flip_labels(labels, poison_y)

    if poison_x > 0.001:
        # For data_poison, replace with random noise as per poison.py implementation when ordered=False
        data, _ = data_poison(data, poison_x, ordered=False)

    # Reconstruct dataset
    poisoned_dataset = MyDataset(torch.from_numpy(data), torch.from_numpy(labels))

    # Use the same batch size as the original dataloader
    batch_size = dataloader.batch_size if dataloader.batch_size is not None else 1

    poisonloader = torch.utils.data.DataLoader(
        poisoned_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)

    return poisonloader