'''
Abstracts DocVQA
'''
from torch.utils.data import Dataset, DataLoader


class DocVQA(Dataset):
    def __init__(self, mode):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self):
        pass

    def get_dataloader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
