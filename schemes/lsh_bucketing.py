



    # Read the hashes for the trajectory
    trajectory_hashes = fh.read_hash_file(file_path)
    
    # Iterate over each layer's hash
    for layer_hash in trajectory_hashes:
        # Convert the list of coordinates into a string
        hash_string = "_".join(map(str, layer_hash))
        
        # Use CityHash for creating a unique key
        hash_key = cityhash.CityHash128(hash_string)
        
        # Place trajectory into the appropriate bucket
        if hash_key not in bucket_system:
            bucket_system[hash_key] = []
        bucket_system[hash_key].append(filename)