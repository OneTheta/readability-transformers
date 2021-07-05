from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class DataReader(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def get_dataloader(self, batch_size: int) -> DataLoader:
        pass
    @abstractmethod
    def get_dataset(self):
        pass
    @abstractmethod
    def __len__(self):  
        pass