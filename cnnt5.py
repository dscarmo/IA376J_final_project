'''
Model that was pre-trained on synthetic images.
'''

import torch
import pytorch_lightning as pl
import numpy as np
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torchvision.transforms import ToTensor

from wikipedia import Wikipedia
from metrics import compute_exact, compute_f1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU())

    def forward(self, x):
        return self.block(x)


class Feature2Embedding(nn.Module):
    '''
    Convert [B, C, H, W] image feature tensor to [B, seq_len, D] (B, 512, 512).
    For backwards compatibility.
    '''
    def __init__(self, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x.permute(0, 2, 3, 1).reshape(-1, 512, int(512*self.scale_factor))


class CNNT5(pl.LightningModule):
    '''
    Custom CNN -> T5 Decoder
    '''
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.eval_transform = ToTensor()
        self.train_transform = self.eval_transform

        if "base" in self.hparams.t5:
            self.scale_factor = 1.5
        else:
            self.scale_factor = 1

        self.embedding_extractor = nn.Sequential(ConvBlock(3, int(16*self.scale_factor)),
                                                 ConvBlock(int(16*self.scale_factor), int(64*self.scale_factor)),
                                                 ConvBlock(int(64*self.scale_factor), int(256*self.scale_factor)),
                                                 ConvBlock(int(256*self.scale_factor), int(512*self.scale_factor)),
                                                 Feature2Embedding(scale_factor=self.scale_factor))

        self.decoder = T5ForConditionalGeneration.from_pretrained(self.hparams.t5)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.t5)

        if not self.hparams.pre_train:
            print(f"Not pre-training, loading trained weight {self.hparams.initial_ckpt}.")
            pre_trained = CNNT5.load_from_checkpoint(self.hparams.initial_ckpt)

            self.embedding_extractor.load_state_dict(pre_trained.embedding_extractor.state_dict())
            self.decoder.load_state_dict(pre_trained.decoder.state_dict())

    def forward(self, batch):
        x, labels, original = batch

        embedding = self.embedding_extractor(x)

        if self.training:
            return self.decoder(encoder_outputs=(embedding,), labels=labels)[0]
        else:
            return self.generate(embedding)

    def extract_features(self, image):
        embedding = self.embedding_extractor(image)

        return self.generate(embedding, generate_hidden_states=True)

    def generate(self, embedding, generate_hidden_states=False):
        max_length = self.hparams.seq_len

        decoded_ids = torch.full((embedding.shape[0], 1),
                                 self.decoder.config.decoder_start_token_id,
                                 dtype=torch.long).to(embedding.device)

        if generate_hidden_states:
            hidden_states = torch.zeros((embedding.shape[0], 512, int(512*self.scale_factor))).to(embedding.device)

        for step in range(max_length):
            output = self.decoder(decoder_input_ids=decoded_ids,
                                  encoder_outputs=(embedding,),
                                  output_hidden_states=generate_hidden_states)

            if generate_hidden_states:
                hidden_states[:, step, :] = output["decoder_hidden_states"][-1].mean(dim=1)

            logits = output['logits']
            next_token_logits = logits[:, -1, :]

            # Greedy decoding
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)

            # Check if output is end of senquence for all batches
            if torch.eq(next_token_id[:, -1], self.tokenizer.eos_token_id).all():
                break

            # Concatenate past ids with new id, keeping batch dimension
            decoded_ids = torch.cat([decoded_ids, next_token_id], dim=-1)

        if generate_hidden_states:
            return decoded_ids, hidden_states
        else:
            return decoded_ids

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('loss', loss, on_epoch=True, on_step=True)
        return loss

    def evaluation_step(self, batch):
        '''
        Same step for validation and testing.
        '''
        _, _, originals = batch

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
        print("Using Wikipedia")
        return Wikipedia("TRAIN", self.tokenizer, transform=self.train_transform).get_dataloader(batch_size=self.hparams.bs, shuffle=True)

    def val_dataloader(self):
        print("Using Wikipedia")
        return Wikipedia("VAL", self.tokenizer, transform=self.eval_transform).get_dataloader(batch_size=self.hparams.bs, shuffle=False)

    def test_dataloader(self):
        return Wikipedia("TEST", self.tokenizer, transform=self.eval_transform).get_dataloader(batch_size=self.hparams.bs, shuffle=False)


if __name__ == "__main__":
    print("Testing pre-trained CNNT5 small...")
    cnn_t5 = CNNT5.load_from_checkpoint("models/wikipedia_pre_train_continue-epoch=1-val_exact_match=0.58-val_f1=0.98.ckpt",
                                        strict=False).eval().cuda()
    hparams = cnn_t5.hparams
    with torch.no_grad():
        output = cnn_t5.extract_features(torch.randn((2, 3, 512, 256)).cuda())

    print(output[1].shape)

    print("Testing CNNT5 base...")
    hparams.t5 = "t5-base"
    cnn_t5 = CNNT5(hparams).eval().cuda()
    with torch.no_grad():
        output = cnn_t5.extract_features(torch.randn((1, 3, 512, 256)).cuda())

    print(output[1].shape)
