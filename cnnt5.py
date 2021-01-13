'''
Model that was pre-trained on synthetic images.
'''
import os
import argparse
import multiprocessing as mp
from sys import argv

import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from tqdm import tqdm
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from dataset import DocVQA
from metrics import compute_exact, compute_f1
from radam import RAdam
from transforms import get_transform


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU())

    def forward(self, x):
        return self.block(x)


class Feature2Embedding(nn.Module):
    '''
    Convert [B, C, H, W] image feature tensor to [B, seq_len, D].
    For backwards compatibility.
    '''
    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, x):
        return x.permute(0, 2, 3, 1).reshape(-1, 512, self.D)


class CNNT5(pl.LightningModule):
    '''
    Custom CNN -> T5 Decoder
    '''
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        if "base" in self.hparams.size:
            self.embedding_extractor = nn.Sequential(ConvBlock(3, 16),
                                                     ConvBlock(16, 64),
                                                     ConvBlock(64, 256),
                                                     ConvBlock(256, 512),
                                                     ConvBlock(512, 768),
                                                     Feature2Embedding(768))
        else:
            self.embedding_extractor = nn.Sequential(ConvBlock(3, 16),
                                                     ConvBlock(16, 64),
                                                     ConvBlock(64, 256),
                                                     ConvBlock(256, 512),
                                                     Feature2Embedding(512))
        print(f"Embedding extractor:\n{self.embedding_extractor}")
        self.decoder = T5ForConditionalGeneration.from_pretrained(self.hparams.size)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.size)

    def forward(self, batch):
        x, labels, original = (batch["document"], batch["target"], batch["target_text"])

        embedding = self.embedding_extractor(x)

        if self.training:
            return self.decoder(encoder_outputs=(embedding,), labels=labels)[0]
        else:
            return self.generate(embedding)

    def extract_features(self, image):
        embedding = self.embedding_extractor(image)

        return self.generate(embedding)

    def generate(self, embedding):
        max_length = self.hparams.tgt_seq_len

        decoded_ids = torch.full((embedding.shape[0], 1),
                                 self.decoder.config.decoder_start_token_id,
                                 dtype=torch.long).to(embedding.device)

        for step in range(max_length):
            output = self.decoder(decoder_input_ids=decoded_ids,
                                  encoder_outputs=(embedding,))

            logits = output['logits']
            next_token_logits = logits[:, -1, :]

            # Greedy decoding
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)

            # Check if output is end of senquence for all batches
            if torch.eq(next_token_id[:, -1], self.tokenizer.eos_token_id).all():
                break

            # Concatenate past ids with new id, keeping batch dimension
            decoded_ids = torch.cat([decoded_ids, next_token_id], dim=-1)

        return decoded_ids

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('loss', loss, on_epoch=True, on_step=True)
        return loss

    def evaluation_step(self, batch):
        '''
        Same step for validation and testing.
        '''
        originals = batch["target_text"]

        pred_token_phrases = self(batch)
        preds = [self.tokenizer.decode(pred_tokens, skip_special_tokens=True) for pred_tokens in pred_token_phrases]

        exact_matches = []
        f1s = []
        for original, pred in zip(originals, preds):
            exact_matches.append(compute_exact(original, pred))
            f1s.append(compute_f1(original, pred))

        exact_match = np.array(exact_matches).mean()
        f1 = np.array(f1s).mean()

        return exact_match, f1

    def validation_step(self, batch, batch_idx):
        exact_match, f1 = self.evaluation_step(batch)

        self.log('val_exact_match', exact_match, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        exact_match, f1 = self.evaluation_step(batch)

        self.log('test_exact_match', exact_match, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_f1', f1, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(DocVQA("train", self.tokenizer, transform=self.hparams.train_transform, no_image=False,
                                 seq_len=self.hparams.seq_len, tgt_seq_len=self.hparams.tgt_seq_len),
                          batch_size=self.hparams.bs, shuffle=True, num_workers=self.hparams.nworkers)

    def val_dataloader(self):
        return DataLoader(DocVQA("val", self.tokenizer, transform=self.hparams.eval_transform, no_image=False,
                                 seq_len=self.hparams.seq_len, tgt_seq_len=self.hparams.tgt_seq_len),
                          batch_size=self.hparams.bs, shuffle=False, num_workers=self.hparams.nworkers)

    def test_dataloader(self):
        return DataLoader(DocVQA("test", self.tokenizer, transform=self.hparams.eval_transform, no_image=False,
                                 seq_len=self.hparams.seq_len, tgt_seq_len=self.hparams.tgt_seq_len),
                          batch_size=self.hparams.bs, shuffle=False, num_workers=self.hparams.nworkers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--size", type=str, default="t5-small", help="Size of model.")
    parser.add_argument("--seq_len", type=int, default=512, help="Transformer sequence length.")
    parser.add_argument("--tgt_seq_len", type=int, default=32, help="Output seq len.")
    parser.add_argument("--lr", type=float, default=5e-5, help="ADAM Learning Rate.")
    parser.add_argument("--bs", type=int, default=32, help="Batch size.")
    parser.add_argument("--acum", type=int, default=1, help="Acum for batch.")
    parser.add_argument("--precision", type=int, default=32, help="Precision.")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=100, help="How many epochs to wait for improvement in validation.")
    parser.add_argument("--transform_str", type=str, default=None, help="String that sets transforms.")
    parser.add_argument("--nworkers", type=object, default=mp.cpu_count(), help="Number of workers to use in dataloading.")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Single word describing experiment.")
    parser.add_argument("--description", type=str, default="No description.", help="Single phrase describing experiment.")
    parser.add_argument("--debug", action="store_true", help="Fast dev run mode.")
    parser.add_argument("--cli_args", type=str, default=str(argv), help="Store command line arguments. Don't change manually.")
    parser.add_argument("--no_fit", action="store_true", help="Do everything except starting the fit.")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU.")
    hparams = parser.parse_args()

    hparams.train_transform, hparams.eval_transform = get_transform(hparams.transform_str)

    print("Hyperparameters")
    for k, v in vars(hparams).items():
        print(f"{k}: {v}")

    if hparams.task == "train":
        model = CNNT5(hparams=hparams)

        if hparams.debug:
            logger = False
            callbacks = None
        else:
            logger = NeptuneLogger(api_key=os.getenv('NEPTUNE_API_TOKEN'),
                                   project_name="dscarmo/layoutlmt5",
                                   experiment_name=hparams.experiment_name,
                                   tags=[hparams.description],
                                   params=vars(hparams))

            dir_path = os.path.join("models", hparams.experiment_name)
            filename = "{epoch}-{val_exact_match:.2f}-{val_f1:.2f}"
            callbacks = [EarlyStopping(monitor="val_f1",
                                       patience=hparams.patience,
                                       verbose=False,
                                       mode='max',
                                       ),
                         ModelCheckpoint(prefix=hparams.experiment_name,
                                         dirpath=dir_path,
                                         filename=filename,
                                         monitor="val_f1",
                                         mode="max")]

        trainer = pl.Trainer(max_epochs=hparams.max_epochs,
                             gpus=0 if hparams.cpu else 1,
                             accumulate_grad_batches=hparams.acum,
                             precision=hparams.precision,
                             logger=logger,
                             callbacks=callbacks,
                             fast_dev_run=hparams.debug,
                             checkpoint_callback=False if hparams.debug else True
                             )

        if not hparams.no_fit:
            trainer.fit(model)
    elif hparams.task == "forward_test":
        print("Testing pre-trained CNNT5 small...")
        cnn_t5 = CNNT5(hparams).eval()
        print("Loaded cnnt5 small.")
        hparams = cnn_t5.hparams
        with torch.no_grad():
            output = cnn_t5((torch.randn((2, 3, 512, 256)),
                             torch.randn((2, 5)),
                             ["aaaaa", "bbbbb"]))

        print(output.shape)

        print("Testing CNNT5 base...")
        hparams.size = "t5-base"
        cnn_t5 = CNNT5(hparams).eval()
        print("Initialized cnnt5 base.")
        with torch.no_grad():
            output = cnn_t5((torch.randn((2, 3, 1024, 512)),
                             torch.randn((2, 5)),
                             ["aaaaa", "bbbbb"]))

        print(output.shape)
