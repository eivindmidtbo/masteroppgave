def is_invalid_hashed_trajectories(layer_i: list[list], layer_j: list[list]) -> bool:
    """Validates the hashed trajectories"""
    if (
        not layer_i
        or all(len(sublist) == 0 for sublist in layer_i)
        or not layer_j
        or all(len(sublist) == 0 for sublist in layer_j)
    ):
        return True
