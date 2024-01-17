import torch
from torch.utils.data import random_split
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from FCN import FCN


class Params:
    def __init__(self):
        self.n_classes = 10
        self.device = 'gpu'
        self.loss_type = "ce"  # "l2" for L2 loss

        self.batch_size = 128
        self.n_epochs = 10
        self.lr = 1e-1
        self.momentum = 0.5

params = Params()


def get_dataloaders(batch_size=params.batch_size):
    full_set = torchvision.datasets.MNIST('.', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    test_set = torchvision.datasets.MNIST('.', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    # create a training and a validation set
    train_set, val_set = random_split(full_set, [55000, 5000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_loss(output, target, loss_type=params.loss_type, n_classes=params.n_classes):
    if loss_type == 'ce':
        # compute Cross Entropy Loss. No need to one hot the target
        loss = F.cross_entropy(output, target)
    elif loss_type == 'l2':
        # compute L2 loss
        target = F.one_hot(target, n_classes).float()
        loss = F.mse_loss(output, target)

    return loss


def train(net, optimizer, train_loader, device):
    net.train()
    pbar = tqdm(train_loader, ncols=100, position=0, leave=True)
    avg_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = get_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_sc = loss.item()
        avg_loss += (loss_sc - avg_loss) / (batch_idx + 1)

        pbar.set_description('train loss: {:.6f} avg loss: {:.6f}'.format(loss_sc, avg_loss))


def valid(net, val_loader, device):
    net.eval()
    val_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = get_loss(output, target)
        val_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    val_loss /= len(val_loader.dataset)
    print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def test(net, test_loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = net(data)
        loss = get_loss(output, target)

        test_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    if params.device != 'cpu' and torch.cuda.is_available():
        device = torch.device("cuda")

    train_loader, val_loader, test_loader = get_dataloaders()

    if params.loss_type == 'l2':
        loss_str = 'L2'
    elif params.loss_type == 'ce':
        loss_str = 'cross entropy'
    else:
        raise AssertionError(f'invalid loss type: {params.loss_type}')

    print(f'\nUsing implementation with {loss_str} loss:')

    net = FCN(params.n_classes).to(device)
    optimizer = optim.SGD(net.parameters(), lr=params.lr,
                          momentum=params.momentum)

    for epoch in range(params.n_epochs):
        print(f'\nepoch {epoch + 1} / {params.n_epochs}\n')
        train(net, optimizer, train_loader, device)

        with torch.no_grad():
            valid(net, val_loader, device)

    with torch.no_grad():
        test(net, test_loader, device)


if __name__ == "__main__":
    main()
