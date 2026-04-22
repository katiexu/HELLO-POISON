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