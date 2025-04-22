from datetime import datetime

import torch
import torchvision.models as models
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from SVHNDataset import SVHNDatset


def main():


    dataset = SVHNDatset('Data/train')


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}



    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    traindata = DataLoader(train_dataset, **params)
    valdata = DataLoader(val_dataset, **params)



    model = models.vgg16(pretrained=True)
    in_features = model.classifier[6].in_features


    model.classifier[6] = nn.Linear(in_features, out_features=11)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)


    # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

    def train_one_epoch(epoch_index, tb_writer):
        model.train()
        running_loss = 0.0
        last_loss = 0.

        for i, data in enumerate(traindata):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()



            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(train_dataset) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss



    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    epochs = 10


    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0

        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(valdata):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == '__main__':
    import torch.multiprocessing
    main()