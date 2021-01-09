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
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMEmbeddings

from dataset import DocVQA
from metrics import compute_exact, compute_f1
from radam import RAdam
from cnnt5 import CNNT5
from transforms import get_transform


class LayoutLMT5(pl.LightningModule):
    '''
    Connect LayoutLM base to T5-base text generator, or
    try to use bounding box embeddings into base t5.
    '''
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.use_radam = getattr(self.hparams, "use_radam", False)

        if not self.hparams.t5_only:
            print("Initializing LayoutLM...")
            self.encoder = LayoutLMModel.from_pretrained(self.hparams.layoutlm_str)
            if self.hparams.freeze_layoutlm:
                for param in tqdm(self.encoder.parameters(), desc="Freezing LayoutLM...", leave=True):
                    param.requires_grad = False

        print("Initializing T5...")
        self.t5 = T5ForConditionalGeneration.from_pretrained(self.hparams.t5_str)
        self.use_llm_emb = getattr(self.hparams, "llm_emb", False)
        if self.use_llm_emb:
            print("Initializing layoutlm embeddings")
            self.llm_emb = LayoutLMEmbeddings(LayoutLMModel.from_pretrained(self.hparams.layoutlm_str).config)
            
        if not self.hparams.no_image:
            print("Using images, CNNT5 based initialized as a image embedding extractor.")
            self.cnnt5 = CNNT5({"t5": self.hparams.t5_str, "pre_train": True,
                                "seq_len": self.hparams.seq_len})
            for param in tqdm(self.cnnt5.parameters(), desc="Freezing CNNT5...", leave=True):
                param.requires_grad = False

        if self.hparams.t5_only:
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.t5_str)
        else:
            self.tokenizer = LayoutLMTokenizer.from_pretrained(self.hparams.layoutlm_str)
        self.detokenizer = T5Tokenizer.from_pretrained(self.hparams.t5_str)

    def my_generate(self, features, max_length=None):
        '''
        Adaptação de código da aula 10 do semestre passado.
        Usa features construídas externamente para gerar frases com T5.
        '''
        if max_length is None:
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
            # LayoutLM features
            features = self.encoder(input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"],
                                    attention_mask=batch["attention_mask"], bbox=batch["bboxes"])[0]
        elif self.use_llm_emb:
            # LayoutLM embeddings, using T5 word embeddings
            t5_embeddings = self.t5.shared(batch["input_ids"])
            features = self.llm_emb(input_ids=batch["input_ids"], bbox=batch["bboxes"], token_type_ids=batch["token_type_ids"],
                                    inputs_embeds=t5_embeddings)
            if not self.hparams.no_image:
                features += self.cnnt5.extract_features(batch["document"])
        else:
            features = None

        if self.training:
            if self.hparams.t5_only:
                if self.use_llm_emb:
                    return self.t5(inputs_embeds=features,
                                   labels=batch["target"])[0]
                else:
                    return self.t5(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   labels=batch["target"])[0]
            else:
                return self.t5(inputs_embeds=features,
                               labels=batch["target"])[0]
        else:
            if self.hparams.t5_only and not self.use_llm_emb:
                return self.t5.generate(input_ids=batch["input_ids"], max_length=32)
            else:
                return self.my_generate(features, max_length=32)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('loss', loss, on_epoch=True, on_step=True)
        return loss

    def evaluation_step(self, batch):
        '''
        Same step for validation and testing.
        '''
        pred_token_phrases = self(batch)
        preds = [self.detokenizer.decode(pred_tokens, skip_special_tokens=True) for pred_tokens in pred_token_phrases]

        return batch["target_text"], preds

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch)

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch)

    def epoch_end(self, outputs, phase):
        tgts, preds = [], []
        for output in outputs:
            tgts += output[0]
            preds += output[1]

        f1s, exacts = [], []
        for tgt, pred in zip(tgts, preds):
            f1s.append(compute_f1(tgt, pred))
            exacts.append(compute_exact(tgt, pred))

        self.log_dict({f"{phase}_f1": np.array(f1s).mean(), f"{phase}_exact_match": np.array(exacts).mean()},
                      prog_bar=True, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self.epoch_end(outputs, "test")

    def configure_optimizers(self):
        if self.use_radam:
            optimizer_str = "RAdam"
            return RAdam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer_str = "Adam"
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        print(f"Optimizer: {optimizer_str}")

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
    parser.add_argument("--llm_emb", action="store_true", help="Use llmembbedings in T5.")
    parser.add_argument("--t5_str", type=str, default="t5-base", help="T5 weights to load.")
    parser.add_argument("--seq_len", type=int, default=512, help="Transformer sequence length.")
    parser.add_argument("--lr", type=float, default=5e-4, help="ADAM Learning Rate.")
    parser.add_argument("--bs", type=int, default=2, help="Batch size.")
    parser.add_argument("--acum", type=int, default=1, help="Acum for batch.")
    parser.add_argument("--precision", type=int, default=32, help="Precision.")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=2, help="How many epochs to wait for improvement in validation.")
    parser.add_argument("--transform_str", type=str, default=None, help="String that sets transforms.")
    parser.add_argument("--nworkers", type=object, default=mp.cpu_count(), help="Number of workers to use in dataloading.")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Single word describing experiment.")
    parser.add_argument("--description", type=str, default="No description.", help="Single phrase describing experiment.")
    parser.add_argument("--no_image", action="store_true", help="Don't load document images.")
    parser.add_argument("--use_radam", action="store_true", help="Use the RADAM optimizer.")
    parser.add_argument("--debug", action="store_true", help="Fast dev run mode.")
    parser.add_argument("--cli_args", type=str, default=str(argv), help="Store command line arguments. Don't change manually.")
    parser.add_argument("--no_fit", action="store_true", help="Do everything except starting the fit.")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Pre trained model to start with.")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU.")
    hparams = parser.parse_args()

    hparams.train_transform, hparams.eval_transform = get_transform(hparams.transform_str)

    if hparams.task == "train":
        model = LayoutLMT5(hparams=hparams)

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

        print("Hyperparameters")
        for k, v in vars(hparams).items():
            print(f"{k}: {v}")

        if not hparams.no_fit:
            trainer.fit(model)
    elif hparams.task == "validate":
        pretrained_model = LayoutLMT5.load_from_checkpoint(hparams.pretrained_model, strict=False)
        validator = pl.Trainer(gpus=1, logger=False)
        validator.test(pretrained_model, test_dataloaders=pretrained_model.val_dataloader())
