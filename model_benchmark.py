import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import get_dataset, get_network, TensorDataset
import copy
import argparse
import time


class GenerateNet(torch.nn.Module):
    def __init__(self, model, channel, num_classes, im_size):
        super().__init__()
        self.network = get_network(model, channel, num_classes, im_size=im_size, dist=True)
        self.output = nn.Linear(13, 1)

    def forward(self, x):
        x = self.network(x)
        x = self.output(x)
        return x.view(-1, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_train = 256

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        'dl-DR17-ttype', '', '256', 'imagenette', args=args)

    images_all = []
    labels_all = []
    print("BUILDING training DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(sample[1])

    images_all = torch.cat(images_all, dim=0).to(device)
    labels_all = torch.tensor(labels_all, dtype=torch.float32).to(device)

    images_test = []
    labels_test = []
    print("BUILDING testing DATASET")
    for i in tqdm(range(len(dst_test))):
        sample = dst_train[i]
        images_test.append(torch.unsqueeze(sample[0], dim=0))
        labels_test.append(sample[1])

    images_test = torch.cat(images_test, dim=0).to(device)
    labels_test = torch.tensor(labels_test, dtype=torch.float32).to(device)

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f' % (
            ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    criterion = nn.MSELoss(reduction='sum').to(device)

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batch_train, shuffle=True, num_workers=0)

    dst_test = TensorDataset(copy.deepcopy(images_test.detach()), copy.deepcopy(labels_test.detach()))
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=batch_train, shuffle=True, num_workers=0)

    train_size = len(images_all)
    test_size = len(images_test)
    for n, epochs in [['galaxy', 30], ['ConvNet', 10]]:
        print('Benchmarking', n, '-------------------------------------------------------------------')
        model = GenerateNet(n, channel=channel, num_classes=num_classes, im_size=im_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        for i in range(epochs):
            losses = 0

            for images, target in tqdm(trainloader):
                pred = model(images)
                outputs = pred.to(device)

                model.zero_grad()
                loss = criterion(outputs, target)
                losses += loss.item()
                loss.backward()
                optimizer.step()

            tests = 0
            for images, target in tqdm(testloader):
                pred = model(images)
                outputs = pred.to(device)
                loss = criterion(outputs, target)
                tests += loss.item()

            print(f"Training loss at epoch {i}: {losses / train_size}, test loss: {tests / test_size}")
            time.sleep(0.1)
