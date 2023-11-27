import config
import training_utils
from clothing_segmentation_dataset import ClothingSementationDataset
from unet import UNet
import torch
# from torchvision.ops.focal_loss import sigmoid_focal_loss
from loss import FocalLoss

device = "cuda" if torch.cuda.is_available() else "mps"
if __name__ == "__main__":
    dataset = ClothingSementationDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    model = UNet().to(device)
    loss_function = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    num_total_steps = len(train_loader)
    num_total_steps_in_test = len(test_loader)

    highest_iou_accuracy = 0.0

    for epoch in range(config.num_epochs):
        print(epoch)
        totalLoss = 0
        total_training_ious_counted = 0
        total_training_iou = 0
        total_training_iou_by_class = torch.tensor([0] * config.number_of_classifications)
        total_training_iou_counted_by_class = torch.tensor([0] * config.number_of_classifications)

        for i, (images, maskImages) in enumerate(train_loader):
            # print(i)
            images = images.to(device).to(torch.float32)
            maskImages = maskImages.to(device).to(torch.float32)
            outputs = model(images)
            loss = loss_function(outputs.to(torch.float), maskImages.to(torch.long))
            totalLoss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_training_ious_counted += 1
            # total_training_iou += training_utils.average_class_iou(outputs, maskImages)

            # ious, counts = training_utils.class_ious(outputs, maskImages)
            # total_training_iou_by_class = total_training_iou_by_class + ious
            # total_training_iou_counted_by_class = total_training_iou_counted_by_class + counts

        totalLoss /= num_total_steps
        average_training_iou = total_training_iou / total_training_ious_counted
        print(f"Epoch: {epoch} Loss: {totalLoss} ")
        print(f"Training IOU: {average_training_iou}")
        print(f"Training IOU By CLASS: {total_training_iou_by_class / total_training_iou_counted_by_class}")
        if epoch % 10 == 0:
            shouldRecord = False

            with torch.no_grad():
                totalCorrect = 0
                numCounted = 0
                total_iou = 0
                total_iou_by_class= torch.tensor([0]*config.number_of_classifications)
                total_iou_counted_by_class = torch.tensor([0]*config.number_of_classifications)
                for j, (test_images, test_masks) in enumerate(test_loader):
                    test_images = test_images.to(device)
                    test_masks = test_masks.to(device)
                    test_output_prob = model(test_images)
                    test_output = test_output_prob.argmax(1)

                    numCorrect = 0
                    totalCorrect += numCorrect
                    numCounted += 1

                    total_iou += training_utils.average_class_iou(test_output_prob,test_masks)
                    ious, counts = training_utils.class_ious(test_output_prob,test_masks)
                    total_iou_by_class = total_iou_by_class + ious
                    total_iou_counted_by_class = total_iou_counted_by_class + counts
                accuracy = totalCorrect/(config.model_image_width * config.model_image_height * numCounted)
                iou_accuracy = total_iou / numCounted
                print(f"IOU Accuracy: {iou_accuracy}")
                print(f"IOU By CLASS: {total_iou_by_class/total_iou_counted_by_class}")
                average_training_iou
                if (average_training_iou >= highest_iou_accuracy or average_training_iou >= .95) and epoch > 0:
                    highest_iou_accuracy = average_training_iou
                    shouldRecord = True
                if shouldRecord:
                    PATH = './saved_models/train_network' + str(epoch) + '.pth'
                    torch.save(model.state_dict(), PATH)
                    print("Saved Model")


