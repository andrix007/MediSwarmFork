import torch
from datamodules.datamodule import SeparateDatasetDataModule

def test_dataloader(name, dataloader):
    print(f"Testing {name} dataloader:")
    for dataset_name, loader in dataloader.items():
        print(f"Dataset: {dataset_name}")
        try:
            for batch_idx, (inputs, labels, paths) in enumerate(loader):
                print(f"Batch {batch_idx}:")
                print(f"  Inputs: {inputs.shape}")
                print(f"  Labels: {labels.shape}")
                print(f"  Paths: {len(paths)}")
                if batch_idx >= 1:  # Test only 2 batches per dataset
                    break
        except Exception as e:
            print(f"Error testing {dataset_name} dataloader: {e}")
        print()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)  # Ensure correct multiprocessing method

    # Initialize the data module
    data_module = SeparateDatasetDataModule(
        cxp_csv_path="/bigdata/andrei_thesis/preprocessed_site_data/CXP/CXP_data.csv",
        cxp_image_dir="/bigdata/andrei_thesis/CXP_data/images",
        mimic_csv_path="/bigdata/andrei_thesis/preprocessed_site_data/MIMIC/MIMIC_data.csv",
        mimic_image_dir="/bigdata/andrei_thesis/MIMIC/mimic-cxr-2.0.0.physionet.org",
        nih_csv_path="/bigdata/andrei_thesis/preprocessed_site_data/NIH/NIH_data.csv",
        nih_image_dir="/bigdata/andrei_thesis/NIH_data/images/images",
        batch_size=4,
        num_workers=2,  # Adjust based on debugging needs
        pin_memory=True,
    )

    print("Setting up the data module...")
    data_module.setup(stage="fit")
    
    # Test training dataloaders
    train_dataloader = data_module.train_dataloader()
    test_dataloader("training", train_dataloader)
