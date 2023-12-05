import config
import training_utils
from clothing_segmentation_dataset import ClothingSementationDataset
from unet import UNet
import torch
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
    model.load_state_dict(torch.load("saved_models/train_network_test.pth"))

    loss_function = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=0)
    num_total_steps = len(train_loader)
    num_total_steps_in_test = len(test_loader)

    highest_iou_accuracy = 0.0

    for epoch in range(config.num_epochs):
        totalLoss = 0
        total_training_ious_counted = 0
        total_training_iou = 0
        total_training_iou_by_class = torch.tensor([0] * config.number_of_classifications)
        total_training_iou_counted_by_class = torch.tensor([0] * config.number_of_classifications)

        for i, (images, maskImages) in enumerate(train_loader):
            images = images.to(device).to(torch.float32)
            maskImages = maskImages.to(device).to(torch.float32)
            outputs = model(images)
            loss = loss_function(outputs.to(torch.float), maskImages.to(torch.long))
            totalLoss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_training_ious_counted += 1

        totalLoss /= num_total_steps
        average_training_iou = total_training_iou / total_training_ious_counted
        print(f"Epoch: {epoch} Loss: {totalLoss} ")
        if epoch % 10 == 0:
            shouldRecord = False
            test_loss = 0
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

                    test_loss = loss_function(test_output_prob.to(torch.float), test_masks.to(torch.long))
                    test_loss += test_loss
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
                print(f"Test Loss:{test_loss / num_total_steps_in_test}")
                print(f"IOU Accuracy: {iou_accuracy}")
                print(f"IOU By CLASS: {total_iou_by_class/total_iou_counted_by_class}")
                
                if (average_training_iou >= highest_iou_accuracy or average_training_iou >= .95) and epoch > 0:
                    highest_iou_accuracy = average_training_iou
                    shouldRecord = True
                
                if shouldRecord:
                    PATH = './saved_models/train_network.pth'
                    torch.save(model.state_dict(), PATH)
                    print("Saved Model")


