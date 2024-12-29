from dataset_cxp import CXP_Dataset
from dataset_mimic import MIMIC_Dataset
from dataset_nih import NIH_Dataset

# Define paths
cxp_csv = "/bigdata/andrei_thesis/preprocessed_site_data/CXP/CXP_training_data.csv"
cxp_images = "/bigdata/andrei_thesis/CXP_data/images"

mimic_csv = "/bigdata/andrei_thesis/preprocessed_site_data/MIMIC/MIMIC_training_data.csv"
mimic_images = "/bigdata/andrei_thesis/MIMIC/mimic-cxr-2.0.0.physionet.org"

nih_csv = "/bigdata/andrei_thesis/preprocessed_site_data/NIH/NIH_training_data.csv"
nih_images = "/bigdata/andrei_thesis/NIH_data/images/images"

# Test CXP
cxp_dataset = CXP_Dataset(csv_path=cxp_csv, image_dir=cxp_images)
print(f"CXP Dataset: {len(cxp_dataset)} samples.")

# Test MIMIC
mimic_dataset = MIMIC_Dataset(csv_path=mimic_csv, image_dir=mimic_images)
print(f"MIMIC Dataset: {len(mimic_dataset)} samples.")

# Test NIH
nih_dataset = NIH_Dataset(csv_path=nih_csv, image_dir=nih_images)
print(f"NIH Dataset: {len(nih_dataset)} samples.")
