import os
import argparse
import multiprocessing as mp
from sys import argv

import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, LayoutLMModel, T5Tokenizer, LayoutLMTokenizer

from dataset import DocVQA
from metrics import compute_exact, compute_f1


class LayoutLMT5(pl.LightningModule):
    '''
    Connect LayoutLM base to T5-base text generator.
    '''
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if not self.hparams.t5_only:
            print("Initializing LayoutLM...")
            self.encoder = LayoutLMModel.from_pretrained(self.hparams.layoutlm_str)
            if self.hparams.freeze_layoutlm:
                for param in tqdm(self.encoder.parameters(), desc="Freezing LayoutLM...", leave=True):
                    param.requires_grad = False

        print("Initializing T5...")
        self.t5 = T5ForConditionalGeneration.from_pretrained(self.hparams.t5_str)

        if self.t5_only:
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.t5_str)
        else:
            self.tokenizer = LayoutLMTokenizer.from_pretrained(self.hparams.layoutlm_str)
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
        if not self.hparams.t5_only:
            # Not working in Transformer 4.0.1
            features = self.encoder(input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"],
                                    attention_mask=batch["attention_mask"], bbox=batch["bboxes"])[0]

        if self.training:
            if self.hparams.t5_only:
                return self.t5(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"],
                               labels=batch["target"])[0]
            else:
                return self.t5(inputs_embeds=features,
                               labels=batch["target"])[0]
        else:
            if self.hparams.t5_only:
                return self.t5.generate(input_ids=batch["input_ids"], max_length=self.hparams.seq_len)
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
        pred_token_phrases = self(batch)
        preds = [self.detokenizer.decode(pred_tokens) for pred_tokens in pred_token_phrases]

        exact_matches = []
        f1s = []
        for original, pred in zip(batch["target_text"], preds):
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
        return DataLoader(DocVQA("train", self.tokenizer, transform=self.hparams.train_transform, no_image=self.hparams.no_image),
                          batch_size=self.hparams.bs, shuffle=True, num_workers=self.hparams.nworkers)

    def val_dataloader(self):
        return DataLoader(DocVQA("val", self.tokenizer, transform=self.hparams.eval_transform, no_image=self.hparams.no_image),
                          batch_size=self.hparams.bs, shuffle=False, num_workers=self.hparams.nworkers)

    def test_dataloader(self):
        return DataLoader(DocVQA("test", self.tokenizer, transform=self.hparams.eval_transform, no_image=self.hparams.no_image),
                          batch_size=self.hparams.bs, shuffle=False, num_workers=self.hparams.nworkers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--layoutlm_str", type=str, default="microsoft/layoutlm-base-uncased", help="LayoutLM weights to load.")
    parser.add_argument("--freeze_layoutlm", action="store_true", help="Freeze layoutlm weights.")
    parser.add_argument("--t5_only", action="store_true", help="Remove LayoutLM from model.")
    parser.add_argument("--t5_str", type=str, default="t5-base", help="T5 weights to load.")
    parser.add_argument("--seq_len", type=int, default=512, help="Transformer sequence length.")
    parser.add_argument("--lr", type=float, default=5e-4, help="ADAM Learning Rate.")
    parser.add_argument("--bs", type=float, default=2, help="Batch size.")
    parser.add_argument("--precision", type=int, default=32, help="Precision.")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=2, help="How many epochs to wait for improvement in validation.")
    parser.add_argument("--train_transform", type=object, default=None, help="Train transform. Can't be set through CLI.")
    parser.add_argument("--eval_transform", type=object, default=None, help="Val and test transform. Can't be set through CLI.")
    parser.add_argument("--nworkers", type=object, default=mp.cpu_count(), help="Number of workers to use in dataloading.")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Single word describing experiment.")
    parser.add_argument("--description", type=str, default="No description.", help="Single phrase describing experiment.")
    parser.add_argument("--no_image", action="store_true", help="Don't load document images.")
    parser.add_argument("--debug", action="store_true", help="Fast dev run mode.")
    parser.add_argument("--cli_args", type=str, default=str(argv), help="Store command line arguments. Don't change manually.")
    parser.add_argument("--no_fit", action="store_true", help="Do everything except starting the fit.")
    hparams = parser.parse_args()

    if hparams.task == "train":
        model = LayoutLMT5(hparams=hparams)

        if hparams.debug:
            callbacks = None
            logger = None
        else:
            neptune_logger = NeptuneLogger(api_key=os.getenv('NEPTUNE_API_TOKEN'),
                                           project_name="dscarmo/layoutlmt5",
                                           experiment_name=hparams.experiment_name,
                                           tags=[hparams.description],
                                           params=vars(hparams))

            early_stopping = EarlyStopping(monitor="val_f1",
                                           patience=hparams.patience,
                                           verbose=False,
                                           mode='max',
                                           )

            dir_path = os.path.join("models", hparams.experiment_name)
            filename = "{epoch}-{val_loss:.2f}-{val_extact_match:.2f}-{val_f1:.2f}"
            checkpoint_callback = ModelCheckpoint(prefix=hparams.experiment_name,
                                                  dirpath=dir_path,
                                                  monitor="val_f1",
                                                  mode="max")

            callbacks = [checkpoint_callback, early_stopping]
            logger = neptune_logger

        trainer = pl.Trainer(max_epochs=hparams.max_epochs,
                             gpus=1,
                             precision=hparams.precision,
                             logger=logger,
                             callbacks=callbacks,
                             fast_dev_run=hparams.debug
                             )

        print("Hyperparameters")
        for k, v in vars(hparams).items():
            print(f"{k}: {v}")

        if not hparams.no_fit:
            trainer.fit(model)
