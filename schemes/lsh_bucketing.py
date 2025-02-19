import cityhash

def place_hashes_into_buckets_original(hashes):
    """Places trajectories into buckets based on their hash values.
    Works for all types of hashes (grid and disk)
    
    Args:    
        hashes (dict): A dictionary containing the hash values of each trajectory.
        
    Returns:
        dict: A dictionary containing the bucket system.
    """

    bucket_system = {}
    # Iterate over each layer's hash
    for hash_file in hashes:
        for layer_hash in hashes[hash_file]:
            
            if layer_hash == []:
                continue

            # Convert the list of coordinates into a string
            hash_string = "_".join(map(str, layer_hash))
            
            # Use CityHash for creating a unique key
            hash_key = cityhash.CityHash128(hash_string)
            
            # Place trajectory into the appropriate bucket
            if hash_key not in bucket_system:
                bucket_system[hash_key] = []
            bucket_system[hash_key].append(hash_file)

    return bucket_system


import cityhash

def place_hashes_into_buckets_loose(hashes):
    """Places trajectories into buckets based on the individual buckets they pass through.
    
    Args:    
        hashes (dict): A dictionary containing the hash values of each trajectory.
        
    Returns:
        dict: A dictionary containing the bucket system.
    """

    bucket_system = {}
    
    # Iterate over each trajectory
    for hash_file in hashes:
        for layer_hash in hashes[hash_file]:
            
            if not layer_hash:
                continue

            # Iterate through each coordinate in the trajectory's layer hash
            for coordinate in layer_hash:
                # Convert the coordinate into a string representation
                coord_string = str(coordinate)
                
                # Hash each coordinate individually
                hash_key = cityhash.CityHash128(coord_string)
                
                # Ensure bucket exists
                if hash_key not in bucket_system:
                    bucket_system[hash_key] = set()  # Use a set to prevent duplicates
                
                # Add trajectory if not already in the bucket
                bucket_system[hash_key].add(hash_file)

    # Convert sets back to lists before returning
    return {key: list(values) for key, values in bucket_system.items()}


