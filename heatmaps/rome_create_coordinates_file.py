import os
import csv
from tqdm import tqdm

# Define the input directory path
input_directory = "dataset/rome/output"

# Define the output file path
output_file = "dataset/rome/rome_all_coordinates.csv"

# Initialize a list to store coordinates
all_coordinates = []

try:
    # Iterate through all files in the directory
    for file_name in tqdm(os.listdir(input_directory), desc="Reading files"):
        # Process only files starting with "R"
        if file_name.startswith("R"):
            file_path = os.path.join(input_directory, file_name)

            # Read the file line by line
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        # Parse latitude and longitude
                        lat, lon = map(float, line.strip().split(","))
                        all_coordinates.append({"latitude": lat, "longitude": lon})
                    except ValueError:
                        # Skip any malformed lines
                        continue

    # Write all coordinates to a CSV file
    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ["latitude", "longitude"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the coordinates
        writer.writerows(all_coordinates)

    print(f"All coordinates have been written to {output_file}")
    print(f"Total coordinates written: {len(all_coordinates)}")

except Exception as e:
    print(f"An error occurred: {e}")
