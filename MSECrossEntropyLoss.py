import torch
import torch.nn as nn
import torch.nn.functional


def distance(target, classes):
    dists = []
    n = target.shape[0]
    for i in range(n):
        cur = torch.arange(-target[i], classes - target[i], 1)
        cur = torch.square(cur)
        cur = torch.div(cur, torch.sum(cur) / (classes - 1))
        cur[target[i]] = 1
        dists.append(cur)
    return torch.stack(dists, 0)


class MSECrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight.to('cuda')
        self.size = weight.shape[0]

    def forward(self, inputs, target):
        output = nn.functional.softmax(inputs, dim=1)
        output = torch.sub(output, 1)
        index = target[:, None].to('cuda')
        mask = torch.ones((inputs.shape[0], 1)).to('cuda')
        output = output.scatter_add_(dim=1, index=index, src=mask)
        output = torch.abs(output)
        output = torch.log(output)
        output = torch.mul(output, self.weight)
        distances = distance(target, inputs.shape[1]).to('cuda')
        output = torch.mul(output, distances)
        loss = torch.sum(output) / 1000

        return -loss


# if __name__ == '__main__':
#     inputs = torch.ones(2,5)
#     target = torch.tensor([1, 2])
#     loss = MSECrossEntropyLoss(torch.ones(5))
#     print(loss(inputs, target))