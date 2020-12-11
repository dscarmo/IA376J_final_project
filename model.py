import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, LayoutLMModel, T5Tokenizer

from dataset import DocVQA
from metrics import compute_exact, compute_f1


class LayoutLMT5(pl.LightningModule):
    '''
    Connect LayoutLM base to T5-base text generator.
    '''
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = LayoutLMModel.from_pretrained(self.hparams.layoutlm_str)
        self.t5 = T5ForConditionalGeneration.from_pretrained(self.hparams.t5_str)
        self.detokenizer = T5Tokenizer.from_pretrained(self.hparams.t5_str)

    def my_generate(self, features):
        '''
        Adaptação de código da aula 10 do semestre passado.
        Usa features construídas externamente para gerar frases com T5.
        '''
        max_length = self.hparams.seq_len

        decoded_ids = torch.full((features.shape[0], 1),
                                 self.t5.config.decoder_start_token_id,
                                 dtype=torch.long).to(features.device)

        encoder_hidden_states = self.t5.get_encoder()(inputs_embeds=features)

        for step in range(max_length):
            outputs = self.t5(decoder_input_ids=decoded_ids,
                              encoder_outputs=encoder_hidden_states,
                              output_attentions=True,
                              return_dict=True)
            logits = outputs["logits"]

            next_token_logits = logits[:, -1, :]

            # Greedy decoding
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)

            # Check if output is end of senquence for all batches
            if torch.eq(next_token_id[:, -1], self.detokenizer.eos_token_id).all():
                break

            # Concatenate past ids with new id, keeping batch dimension
            decoded_ids = torch.cat([decoded_ids, next_token_id], dim=-1)

        return decoded_ids

    def forward(self, batch):
        x, labels, original = batch

        # LayoutLM features
        features = self.encoder.extract_features(x)

        # Decode features
        if self.training:
            # Return will be loss already
            return self.t5(inputs_embeds=features,
                           labels=labels)[0]
        else:
            return self.my_generate(features)

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
        preds = [self.detokenizer.decode(pred_tokens) for pred_tokens in pred_token_phrases]

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
        return DataLoader(DocVQA("train", transform=self.hparams.train_transform),
                          batch_size=self.hparams.bs, shuffle=True, num_workers=self.hparams.nworkers)

    def val_dataloader(self):
        return DataLoader(DocVQA("val", transform=self.hparams.eval_transform),
                          batch_size=self.hparams.bs, shuffle=False, num_workers=self.hparams.nworkers)

    def test_dataloader(self):
        return DataLoader(DocVQA("test", transform=self.hparams.eval_transform),
                          batch_size=self.hparams.bs, shuffle=False, num_workers=self.hparams.nworkers)
