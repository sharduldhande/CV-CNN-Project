#heavily inspired by https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
import pickle

import torch
import os


class Datset(torch.utils.data.Dataset):
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
                self.bboxes.append(bboxes)
                self.names.append(name)




    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, index):

        bbox = self.bboxes[index]
        name = self.names[index]
        image_path = os.path.join(self.image_dir, name)






