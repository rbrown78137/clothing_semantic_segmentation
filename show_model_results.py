from unet import UNet
import torch
import cv2 as cv
import config
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "mps"

if __name__ == "__main__":
    model = UNet().to(device)
    model.load_state_dict(torch.load("saved_models/train_network_test.pth"))

    for idx in range(1000):
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

        image = image.to(device).to(torch.float32).unsqueeze(0)
        model_prediction = model(image)
        outputs = model(image).squeeze(0)
        outputs = outputs.argmax(0)
		
        image_tensor = image[0].detach()
        plt.subplot(1, 2, 1)
        plt.imshow(image_tensor.to("cpu").permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(outputs.to("cpu")/256)
        breakpoint()