from datamodule import NIHDataModule

# In your DataModule or a separate script
def check_val_dataset_size(data_module):
    val_loader = data_module.val_dataloader()
    total_samples = 0
    for batch in val_loader:
        imgs, labels = batch[:2]
        total_samples += imgs.size(0)
    print(f"Total validation samples: {total_samples}")

# Call this function
if __name__ == "__main__":
    check_val_dataset_size(NIHDataModule)
