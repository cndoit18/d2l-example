import torchvision

from torch.utils import data
from torchvision import transforms
import os


def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"),
        train=True,
        transform=trans,
        download=True,
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"),
        train=False,
        transform=trans,
        download=True,
    )
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
        data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4),
    )


if __name__ == "__main__":
    load_data_fashion_mnist(32, resize=64)
