import torch
from torch.utils.data import Dataset
import cv2 as cv
import os
import matplotlib.pyplot as plt
import config

class ClothingSementationDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.number_of_images = len([name for name in os.listdir('./data/png_images/IMAGES')])
        self.images = []
        self.masks = []
        for idx in range(self.number_of_images):
            image,mask = self.get_image_mask(idx)
            self.images.append(image)
            self.masks.append(mask)

    def __len__(self):
        return self.number_of_images

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]
    
    def get_image_mask(self,idx):
        image_path = f'data/png_images/IMAGES/img_{idx+1:04d}.png'
        mask_path = f'data/png_masks/MASKS/seg_{idx+1:04d}.png'
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.resize(image, (config.model_image_width, config.model_image_height))
        image = torch.from_numpy(image).to(torch.float)
        image = image.permute(2, 0, 1)
        image = image / 256

        mask = cv.cvtColor(cv.imread(mask_path, cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY)
        mask = cv.resize(mask, (config.model_image_width, config.model_image_height))
        mask = torch.from_numpy(mask).to(torch.long)

        # Remove skin and hair labels
        mask = torch.where(mask == 19, 0, mask)
        mask = torch.where(mask == 41, 0, mask)

        # Remove Unneccesary attributes
        
        # Combine Attributes

        return image, mask

if __name__ == "__main__":
    dataset = ClothingSementationDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        pass