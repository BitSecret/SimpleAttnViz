from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
from utils import get_config, set_seed
import matplotlib.pyplot as plt

config = get_config()


def split_image(image):
    """
    Split an image into patches.
    :param image: <class 'torch.Tensor'>, torch.Size([28, 28])
    :return: <class 'torch.Tensor'>, torch.Size([49, 16])
    """
    C, H, W = image.shape
    h_block, w_block = config["data"]["patch_size"]

    # 检查是否能整除
    assert H % h_block == 0, "Height must be divisible by patch height"
    assert W % w_block == 0, "Width must be divisible by patch width"

    # 计算分块数量
    n_h = H // h_block
    n_w = W // w_block

    # 分块并展平（关键步骤）
    # Shape: (N_patches, C * h_block * w_block)
    patches = image.reshape(C, n_h, h_block, n_w, w_block).transpose(1, 2).reshape(C * h_block * w_block, n_h * n_w).T

    return patches


class MyDataset(Dataset):
    def __init__(self, data_type, raw_data):
        self.data_type = data_type
        self.raw_data = raw_data  # (image, label)
        self.data = []  # (image, label)

        for image, label in self.raw_data:
            image = split_image(image)
            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def show_raw_data(self):
        print(f"data_type: '{self.data_type}'")
        print(f"len(data): '{len(self.raw_data)}'")
        print(f"type(image): {type(self.raw_data[0][0])}")  # <class 'torch.Tensor'>,
        print(f"image.shape: {self.raw_data[0][0].shape}")  # torch.Size([1, 28, 28]), value in [0, 1]
        print(f"type(label): {type(self.raw_data[0][1])}")
        print(f"label: {self.raw_data[0][1]}")
        plt.imshow(self.raw_data[0][0].squeeze(), cmap='gray')
        plt.title(f"Label: {self.raw_data[0][1]}")
        plt.axis('off')
        plt.show()

    def show_data(self):
        print(f"data_type: '{self.data_type}'")
        print(f"len(data): '{len(self.data)}'")
        print(f"type(image): {type(self.data[0][0])}")  # <class 'torch.Tensor'>,
        print(f"image.shape: {self.data[0][0].shape}")  # torch.Size([49, 16]), value in [0, 1]
        print(f"type(label): {type(self.data[0][1])}")
        print(f"label: {self.data[0][1]}")


def get_datasets():
    transform = transforms.Compose([transforms.ToTensor()])  # To 0~1

    train_data = datasets.MNIST(  # 60000
        root='../../data',
        train=True,
        download=True,  # 如果本地不存在则自动下载
        transform=transform
    )

    test_data = datasets.MNIST(  # 10000
        root='../../data',
        train=False,
        download=True,
        transform=transform
    )

    val_size = int(0.5 * len(test_data))  # 50% 作为验证集，剩余作为测试集
    test_size = len(test_data) - val_size

    val_data, test_data = random_split(test_data, [val_size, test_size])  # 随机拆分

    train_dataset = MyDataset("train", train_data)  # 60000
    val_dataset = MyDataset("val", val_data)  # 5000
    test_dataset = MyDataset("test", test_data)  # 5000

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    set_seed()
    train, val, test = get_datasets()
    train.show_data()
