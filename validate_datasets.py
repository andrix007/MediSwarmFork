import os
import pandas as pd

def validate_dataset(metadata_path, image_dir, file_column_name, file_extension=".dcm"):
    """
    Validates the dataset by checking for missing or inaccessible files.
    
    Args:
        metadata_path (str): Path to the metadata CSV file.
        image_dir (str): Path to the directory containing the image files.
        file_column_name (str): Column in the metadata containing file paths.
        file_extension (str): Expected file extension for the images.
    
    Returns:
        None: Prints missing files and summary.
    """
    # Load metadata
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"Loaded metadata: {len(metadata)} entries found.")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    missing_files = []
    inaccessible_files = []
    valid_files = 0

    # Check each file
    for idx, row in metadata.iterrows():
        file_name = row[file_column_name]
        file_path = os.path.join(image_dir, file_name)

        if not file_name.endswith(file_extension):
            print(f"Warning: Unexpected file extension in metadata: {file_name}")
        
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        elif not os.access(file_path, os.R_OK):
            inaccessible_files.append(file_path)
        else:
            valid_files += 1

    # Summary
    print("\nValidation Summary:")
    print(f"Valid files: {valid_files}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Inaccessible files: {len(inaccessible_files)}")
    
    if missing_files:
        with open("missing_files.txt", "w") as f:
            f.writelines("\n".join(missing_files))
        print("Missing files written to 'missing_files.txt'")
    
    if inaccessible_files:
        with open("inaccessible_files.txt", "w") as f:
            f.writelines("\n".join(inaccessible_files))
        print("Inaccessible files written to 'inaccessible_files.txt'")


if __name__ == "__main__":
    validate_dataset(
        metadata_path="/bigdata/andrei_thesis/preprocessed_site_data/NIH/nih_val_data.csv",
        image_dir="/bigdata/andrei_thesis/NIH_data/images",
        file_column_name="Jointpath",
        file_extension=".png"
    )
