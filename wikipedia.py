import random
import textwrap
import multiprocessing as mp

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import ftfy
import torch
import numpy as np
import pytorch_lightning as pl
import transformers
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datasets import load_dataset

print(f"PL version: {pl.__version__}\npytorch version: {torch.__version__}\nTransformers version: {transformers.__version__}")


class Wikipedia(Dataset):
    '''
    Coloca as 128 primeiras palavras em uma imagem com fundo branco, com augmentations.
    '''
    def __init__(self, mode, tokenizer, seq_len=512, width=256, height=512, transform=None):
        '''
        mode: um de "TRAIN", "VAL", "TEST".
        seq_len: tamanho máximo de sequência. 128 padrão para alinhar com feature 16x8 da efficientnet.
        transform: transformadas para serem aplicadas somente na imagem.
        '''
        super().__init__()
        assert mode in ["TRAIN", "VAL", "TEST"]
        self.mode = mode
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.width = width
        self.height = height
        self.transform = transform
        self.dataset = load_dataset('wikipedia', '20200501.en')["train"]

        # 0.8 0.1 0.1 hold-out split
        dataset_range = range(len(self.dataset))

        train_range, test_range = train_test_split(dataset_range, train_size=0.8, random_state=4321, shuffle=True)
        val_range, test_range = train_test_split(test_range, test_size=0.5)

        self.idx_range = {"TRAIN": train_range, "VAL": val_range, "TEST": test_range}

    def __len__(self):
        return len(self.idx_range[self.mode])

    def __getitem__(self, i):
        '''
        Imagens são geradas a partir do caption
        Transformadas padrão são normalização para efficientnet com advprop e totensor.
        '''
        try:
            idx = self.idx_range[self.mode][i]

            # Tries to fix possible encoding errors and take only first 128 words from wikipedia texts.
            original = ' '.join(ftfy.fix_text(self.dataset[idx]["text"].encode('ascii', errors='ignore').decode()).split()[:128])

            if self.mode == "TRAIN":
                target = self.tokenizer.encode(original,
                                               padding='max_length',
                                               truncation=True,
                                               max_length=self.seq_len,
                                               return_tensors='pt')[0]
            else:
                # Avoid tokenizing computational cost when not training.
                target = original

            image = self.text_to_image(text=original, max_width=self.width, max_height=self.height)

            if self.transform is not None:
                image = self.transform(image)

            return image, target, original
        except Exception as e:
            print(f"WARNING: Error in dataset: {e}\nSkipping {i}th item from {self.mode} dataset.")
            return self[i + 1]

    def get_dataloader(self, batch_size, shuffle, num_workers=mp.cpu_count()):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    def text_to_image(self, text: str, max_width: int, max_height: int):
        '''
        Puts text on image. Notice there are many random parameters for augmentation.
        '''
        # Font size augmentation
        font = ImageFont.truetype("arial.ttf", random.randint(10, 14))

        # Wrap point augmentation
        text = '\n'.join(textwrap.wrap(text, random.randint(30, 32)))

        image = Image.new('RGB', (max_width, max_height), (255, 255, 255))
        d = ImageDraw.Draw(image)

        # Text position augmentation
        d.text((random.randint(10, 20), random.randint(10, 20)), text, font=font, fill=(0, 0, 0))

        text_width, text_height = d.textsize(text)

        return np.array(image)
