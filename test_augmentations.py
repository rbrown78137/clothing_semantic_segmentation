# import torch
# import matplotlib.pyplot as plt
# import cv2 as cv
# import config
# import augmentations
# rows = 8
# columns = 2
# for idx in range(0, 200):
#     fig = plt.figure()

#     bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/'+str(idx+1)+".png", cv.IMREAD_COLOR)
#     unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
#     camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
#     camera_tensor = torch.from_numpy(camera_image)
#     camera_tensor = camera_tensor.permute(2, 0, 1)
#     fig.add_subplot(rows, columns, 1)
#     plt.imshow(camera_tensor.permute(1, 2, 0))
#     plt.axis('off')

#     unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx+1)+".png", cv.IMREAD_COLOR),cv.COLOR_BGR2GRAY)
#     label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height),interpolation=cv.INTER_NEAREST)
#     label_tensor = torch.from_numpy(label_image)
#     #combine labels
#     label_tensor = label_tensor.to(torch.long)
#     label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
#     label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
#     label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
#     label_tensor = label_tensor.to(torch.uint8)
#     fig.add_subplot(rows, columns, 2)
#     plt.imshow(label_tensor)
#     plt.axis('off')

#     bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/' + str(idx + 1) + ".png", cv.IMREAD_COLOR)
#     bgr_image = augmentations.horizontal_shift_reflect(bgr_image, 0.5)
#     unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
#     camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
#     camera_tensor = torch.from_numpy(camera_image)
#     camera_tensor = camera_tensor.permute(2, 0, 1)
#     fig.add_subplot(rows, columns, 3)
#     plt.imshow(camera_tensor.permute(1, 2, 0))
#     plt.axis('off')

#     unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx + 1) + ".png", cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY)
#     unmodified_label_image = augmentations.horizontal_shift_reflect(unmodified_label_image, 0.5)
#     label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height), interpolation=cv.INTER_NEAREST)
#     label_tensor = torch.from_numpy(label_image)
#     # combine labels
#     label_tensor = label_tensor.to(torch.long)
#     label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
#     label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
#     label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
#     label_tensor = label_tensor.to(torch.uint8)
#     fig.add_subplot(rows, columns, 4)
#     plt.imshow(label_tensor)
#     plt.axis('off')

#     bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/' + str(idx + 1) + ".png", cv.IMREAD_COLOR)
#     bgr_image = augmentations.vertical_shift_reflect(bgr_image, 0.5)
#     unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
#     camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
#     camera_tensor = torch.from_numpy(camera_image)
#     camera_tensor = camera_tensor.permute(2, 0, 1)
#     fig.add_subplot(rows, columns, 5)
#     plt.imshow(camera_tensor.permute(1, 2, 0))
#     plt.axis('off')

#     unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx + 1) + ".png", cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY)
#     unmodified_label_image = augmentations.vertical_shift_reflect(unmodified_label_image, 0.5)
#     label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height), interpolation=cv.INTER_NEAREST)
#     label_tensor = torch.from_numpy(label_image)
#     # combine labels
#     label_tensor = label_tensor.to(torch.long)
#     label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
#     label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
#     label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
#     label_tensor = label_tensor.to(torch.uint8)
#     fig.add_subplot(rows, columns, 6)
#     plt.imshow(label_tensor)
#     plt.axis('off')

#     bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/' + str(idx + 1) + ".png", cv.IMREAD_COLOR)
#     bgr_image = augmentations.horizontal_shift_resize(bgr_image, 0.5)
#     unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
#     camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
#     camera_tensor = torch.from_numpy(camera_image)
#     camera_tensor = camera_tensor.permute(2, 0, 1)
#     fig.add_subplot(rows, columns, 7)
#     plt.imshow(camera_tensor.permute(1, 2, 0))
#     plt.axis('off')

#     unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx + 1) + ".png", cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY)
#     unmodified_label_image = augmentations.horizontal_shift_resize(unmodified_label_image, 0.5)
#     label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height), interpolation=cv.INTER_NEAREST)
#     label_tensor = torch.from_numpy(label_image)
#     # combine labels
#     label_tensor = label_tensor.to(torch.long)
#     label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
#     label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
#     label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
#     label_tensor = label_tensor.to(torch.uint8)
#     fig.add_subplot(rows, columns, 8)
#     plt.imshow(label_tensor)
#     plt.axis('off')

#     bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/' + str(idx + 1) + ".png", cv.IMREAD_COLOR)
#     bgr_image = augmentations.vertical_shift_resize(bgr_image, 0.5)
#     unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
#     camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
#     camera_tensor = torch.from_numpy(camera_image)
#     camera_tensor = camera_tensor.permute(2, 0, 1)
#     fig.add_subplot(rows, columns, 9)
#     plt.imshow(camera_tensor.permute(1, 2, 0))
#     plt.axis('off')

#     unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx + 1) + ".png", cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY)
#     unmodified_label_image = augmentations.vertical_shift_resize(unmodified_label_image, 0.5)
#     label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height), interpolation=cv.INTER_NEAREST)
#     label_tensor = torch.from_numpy(label_image)
#     # combine labels
#     label_tensor = label_tensor.to(torch.long)
#     label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
#     label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
#     label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
#     label_tensor = label_tensor.to(torch.uint8)
#     fig.add_subplot(rows, columns, 10)
#     plt.imshow(label_tensor)
#     plt.axis('off')

#     bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/' + str(idx + 1) + ".png", cv.IMREAD_COLOR)
#     bgr_image = augmentations.horizontal_flip(bgr_image)
#     unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
#     camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
#     camera_tensor = torch.from_numpy(camera_image)
#     camera_tensor = camera_tensor.permute(2, 0, 1)
#     fig.add_subplot(rows, columns, 11)
#     plt.imshow(camera_tensor.permute(1, 2, 0))
#     plt.axis('off')

#     unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx + 1) + ".png", cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY)
#     unmodified_label_image = augmentations.horizontal_flip(unmodified_label_image)
#     label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height), interpolation=cv.INTER_NEAREST)
#     label_tensor = torch.from_numpy(label_image)
#     # combine labels
#     label_tensor = label_tensor.to(torch.long)
#     label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
#     label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
#     label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
#     label_tensor = label_tensor.to(torch.uint8)
#     fig.add_subplot(rows, columns, 12)
#     plt.imshow(label_tensor)
#     plt.axis('off')

#     bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/' + str(idx + 1) + ".png", cv.IMREAD_COLOR)
#     bgr_image = augmentations.cropped_rotated_image(bgr_image, 30)
#     unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
#     camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
#     camera_tensor = torch.from_numpy(camera_image)
#     camera_tensor = camera_tensor.permute(2, 0, 1)
#     fig.add_subplot(rows, columns, 13)
#     plt.imshow(camera_tensor.permute(1, 2, 0))
#     plt.axis('off')

#     unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx + 1) + ".png", cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY)
#     unmodified_label_image = augmentations.cropped_rotated_image(unmodified_label_image,30)
#     label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height), interpolation=cv.INTER_NEAREST)
#     label_tensor = torch.from_numpy(label_image)
#     # combine labels
#     label_tensor = label_tensor.to(torch.long)
#     label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
#     label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
#     label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
#     label_tensor = label_tensor.to(torch.uint8)
#     fig.add_subplot(rows, columns, 14)
#     plt.imshow(label_tensor)
#     plt.axis('off')

#     bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/' + str(idx + 1) + ".png", cv.IMREAD_COLOR)
#     bgr_image = augmentations.zoom(bgr_image, 0.5)
#     unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
#     camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
#     camera_tensor = torch.from_numpy(camera_image)
#     camera_tensor = camera_tensor.permute(2, 0, 1)
#     fig.add_subplot(rows, columns, 15)
#     plt.imshow(camera_tensor.permute(1, 2, 0))
#     plt.axis('off')

#     unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx + 1) + ".png", cv.IMREAD_COLOR), cv.COLOR_BGR2GRAY)
#     unmodified_label_image = augmentations.zoom(unmodified_label_image, 0.5)
#     label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height), interpolation=cv.INTER_NEAREST)
#     label_tensor = torch.from_numpy(label_image)
#     # combine labels
#     label_tensor = label_tensor.to(torch.long)
#     label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
#     label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
#     label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
#     label_tensor = label_tensor.to(torch.uint8)
#     fig.add_subplot(rows, columns, 16)
#     plt.imshow(label_tensor)
#     plt.axis('off')

#     breakSpot = 1