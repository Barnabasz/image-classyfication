import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Image_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, images_list, root_dir, transform=transforms.ToTensor()):
        'Initialization'
        self.images_list = images_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        image_name = self.images_list[index][0]
        label = self.images_list[index][1]
        image_name = os.path.join(self.root_dir, image_name)
        image = Image.open(image_name)
        image = self.transform(image)
        return image, label
    
    @staticmethod
    def load_images(directory_in_str, classes={}):
        directory = os.fsencode(directory_in_str)
        images_list = []
        for class_dir in os.listdir(directory):
            classes.setdefault(os.fsdecode(class_dir),len(classes))
            for image in os.listdir(os.path.join(directory, class_dir)):
                images_list.append((os.fsdecode(os.path.join(class_dir, image)), torch.tensor(classes[os.fsdecode(class_dir)])))
        return (images_list, classes)

if __name__ == "__main__":
    directory_in_str = "111880_269359_upload_seg_train/seg_train"
    images_list, classes = Image_Dataset.load_images(directory_in_str)
    print(images_list[:10])
    print(classes)
    dataset = Image_Dataset(images_list, directory_in_str, transforms.ToTensor())
    print(dataset[10][0].shape)