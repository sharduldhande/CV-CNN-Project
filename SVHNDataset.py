#heavily inspired by https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

#particulars of working with vgg16 taken from https://medium.com/@piyushkashyap045/transfer-learning-in-pytorch-fine-tuning-pretrained-models-for-custom-datasets-6737b03d6fa2

import pickle
import torch
import os
import cv2
from torchvision.transforms import transforms
from PIL import Image


class SVHNDatset(torch.utils.data.Dataset):
    def __init__(self, image_dir):

        self.image_dir = image_dir
        bboxes_path = 'bboxes.pkl'
        names_path = 'names.pkl'
        nondigiboxes = 'nondigiboxes.pkl'
        nondiginames = 'nondiginames.pkl'

        bboxes_path = os.path.join(image_dir, bboxes_path)
        names_path = os.path.join(image_dir, names_path)
        nondigiboxes = os.path.join(image_dir, nondigiboxes)
        nondiginames = os.path.join(image_dir, nondiginames)

        with open(bboxes_path, 'rb') as file:
            bboxeslist = pickle.load(file)
        with open(names_path, 'rb') as file1:
            nameslist = pickle.load(file1)
        with open(nondigiboxes, 'rb') as file2:
            bboxeslist.extend(pickle.load(file2))
        with open(nondiginames, 'rb') as file3:
            nameslist.extend(pickle.load(file3))

        self.bboxes = []
        self.names = []

        for i in range(len(bboxeslist)):
            abboxes = bboxeslist[i]
            name = nameslist[i]

            for bbox in abboxes:
                self.bboxes.append(bbox)
                self.names.append(name)




    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, index):

        bbox = self.bboxes[index]
        name = self.names[index]
        image_path = os.path.join(self.image_dir, name)
        x2 = bbox['left']
        y2 = bbox['top']
        w2 = bbox['width']
        h2 = bbox['height']
        label = bbox['label']
        # image = cv2.imread(image_path)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # slice = image[y2:y2+h2, x2:x2+w2]

        #since torchvision expects PIL, opening image with PIL directly

        image = Image.open(image_path).convert('RGB')
        slice = image.crop((x2,y2,x2+w2,y2+h2))



        #https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        #https://medium.com/@piyushkashyap045/transfer-learning-in-pytorch-fine-tuning-pretrained-models-for-custom-datasets-6737b03d6fa2

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        slice1 = transform(slice)

        return slice1, label

