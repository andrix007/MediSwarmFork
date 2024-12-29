from nvflare.app_common.pt.pt_fed_client import PTFedClient
from custom.models.model_lit import LitModel
from custom.data.datamodules.datamodule import LitDataModule

def main():
    data_module = LitDataModule(
        train_csv="/path/to/train.csv",
        val_csv="/path/to/val.csv",
        test_csv="/path/to/test.csv",
        image_dir="/path/to/images",
        batch_size=32
    )

    model = LitModel(lr=0.0005, criterion_name="BCELoss", num_labels=8)

    fl_client = PTFedClient(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader()
    )

    fl_client.start()

if __name__ == "__main__":
    main()
