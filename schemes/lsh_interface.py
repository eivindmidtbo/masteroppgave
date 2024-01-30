"""
Superclass for LSHschemes
"""


class LSHInterface:
    """Interface for LSH classes"""

    def _create_trajectory_hash(self, trajectory: list[list[float]]) -> list[list[str]]:
        """Hash a single trajectory"""
        pass

    def compute_dataset_hashes(self) -> dict[str, list]:
        """Compute all trajectory hashes"""
        pass

    def measure_hash_computation(self, number: int, repeat: int) -> None:
        """Method for measuring the actual compuation time needed to hash"""
        pass

    def print_hashes(self) -> None:
        """Method to print the schemes hashes"""
        pass

    def set_meta_file(self, meta_file: str) -> None:
        """Method to set a schemes meta_file"""
        pass
