import numpy as np
from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
    import numpy as np

    mp = np.array(matrix_profile["mp"], dtype=float)
    pi = np.array(matrix_profile["mpi"], dtype=int)

    # Filter out invalid entries (NaN or Inf)
    valid_indices = np.isfinite(mp)
    valid_positions = np.where(valid_indices)[0]
    valid_mp = mp[valid_indices]
    valid_pi = pi[valid_indices]

    # Sort the valid matrix profile values in descending order
    sorted_order = np.argsort(valid_mp)[::-1]

    # Adjust top_k if there are fewer valid entries
    top_k = min(top_k, len(sorted_order))

    # Select top-k discords
    top_k_indices_local = sorted_order[:top_k]
    top_k_indices = valid_positions[top_k_indices_local]
    discord_distances = mp[top_k_indices]
    nearest_neighbor_indices = pi[top_k_indices]

    # Construct the result dictionary
    discords = {
        "discord_indices": top_k_indices,
        "discord_distances": discord_distances,
        "nearest_neighbor_indices": nearest_neighbor_indices,
    }
    return {
        "indices": top_k_indices,
        "distances": discord_distances,
        "nn_indices": nearest_neighbor_indices,
    }
