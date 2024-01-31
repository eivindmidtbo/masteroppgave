class Disk():
    """Class structure for disk-based LSH"""
    def __init__(self, name: int, lat: float, lon: float) -> None:
        self.name = name
        self.lat = lat
        self.lon = lon
