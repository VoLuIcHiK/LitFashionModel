import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

PATH_DATASETS = './lesson4'
BATCH_SIZE = 64


class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, learning_rate, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.learning_rate = learning_rate

    def prepare_data(self):
        #скачивание данных
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        #разделение на train/val для использования даталоэдером
        if stage == "fit" or stage is None:
            fmnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.fmnist_train, self.fmnist_val = random_split(fmnist_full, [55000, 5000])

        #тестовый датасет для использования даталоэдером
        if stage == "test" or stage is None:
            self.fmnist_test = FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.fmnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.fmnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.fmnist_test, batch_size=BATCH_SIZE)
