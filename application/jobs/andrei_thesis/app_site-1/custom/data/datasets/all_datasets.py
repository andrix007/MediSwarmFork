import torch
from torch.utils.data import Dataset
import os
import numpy as np
from imageio import imread
from PIL import Image
from torchvision import transforms


class AllDatasetsShared(Dataset):
    def __init__(self, dataframe, finding="any", transform=None):
        """
        Dataset class for handling aggregated datasets (CXP, MIMIC, NIH).

        Arguments:
        dataframe: DataFrame containing dataset metadata.
        finding: Specific finding to target (default: "any").
        transform: Transformations to apply to the images.
        """
        self.dataframe = dataframe
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),  # Ensures uniform image sizes
            transforms.ToTensor()
        ])
        self.PRED_LABEL = [
            'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion',
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
        ]

    def __getitem__(self, idx):
        while True:  # Retry until a valid image is found
            item = self.dataframe.iloc[idx]
            try:
                # Read the image
                img = imread(item["Jointpath"])
                
                # Ensure image has the correct shape
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)
                elif len(img.shape) == 3 and img.shape[2] > 3:  # Extra channels
                    img = img[:, :, :3]
                
                # Convert to uint8 if necessary
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                
                # Convert to PIL image
                img = Image.fromarray(img)
                
                # Apply transformations
                if self.transform:
                    img = self.transform(img)
                
                # Generate labels
                label = torch.zeros(len(self.PRED_LABEL), dtype=torch.float32)
                for i, pred_label in enumerate(self.PRED_LABEL):
                    value = item[pred_label.strip()]
                    if not np.isnan(value):
                        label[i] = float(value)

                return img, label, item["Jointpath"]

            except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
                print(f"Error loading image {item['Jointpath']}: {e}")
                idx = (idx + 1) % self.dataset_size  # Move to next index

    def __len__(self):
        return self.dataset_size
