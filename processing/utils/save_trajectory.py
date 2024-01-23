# Some helper functions for controling and saving the trajectories


# Save function
def save_current_trajectory(
    OUTPUT_FOLDER: str, file_name: str, trajectory: list[tuple[float]]
) -> None:
    with open(f"{OUTPUT_FOLDER}/R_{file_name}.txt", "w") as file:
        for coordinate in trajectory:
            lat, lon = coordinate
            file.write("%s, %s\n" % (lat, lon))
        file.close()
        return
