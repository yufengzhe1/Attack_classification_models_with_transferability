from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
import csv
import numpy as np
import os


transforms = transforms.Compose([transforms.ToTensor()])


# class of dataset, load the images from given path and label file
class MyDataset(Dataset):
    def __init__(self, csv_path, path, transform=transforms):
        super(MyDataset, self).__init__()
        images = []
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                images.append((path + str(row['ImageId']), int(row['TrueLabel'])))

        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        filename, label = self.images[index]
        img = Image.open(filename)
        img = self.transform(img)
        return img, label, filename

    def __len__(self):
        return len(self.images)


# standard imagenet normalize
class imgnormalize(nn.Module):
    def __init__(self):
        super(imgnormalize, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


norm = imgnormalize()  # standard imagenet normalize


# save adv images to result folder
def save_imgs(X, adv_img_save_folder, filenames):
    for i in range(X.shape[0]):
        adv_final = X[i].cpu().detach().numpy()
        adv_final = (adv_final*255).astype(np.uint8)
        file_path = os.path.join(adv_img_save_folder, filenames[i].split('/')[-1])
        adv_x_255 = np.transpose(adv_final, (1, 2, 0))
        im = Image.fromarray(adv_x_255)
        # quality can be affects the robustness of the adversarial images
        im.save(file_path, quality=99)


