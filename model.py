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
        # LayoutLM features
        features = self.encoder(input_ids=batch["input_tokens"], bbox=batch["bbox"])[0]

        # Decode features
        if self.training:
            # Return will be loss already
            return self.t5(inputs_embeds=features,
                           labels=batch["target"])[0]
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
        return DataLoader(DocVQA("train", transform=self.hparams.train_transform),
                          batch_size=self.hparams.bs, shuffle=True, num_workers=self.hparams.nworkers)

    def val_dataloader(self):
        return DataLoader(DocVQA("val", transform=self.hparams.eval_transform),
                          batch_size=self.hparams.bs, shuffle=False, num_workers=self.hparams.nworkers)

    def test_dataloader(self):
        return DataLoader(DocVQA("test", transform=self.hparams.eval_transform),
                          batch_size=self.hparams.bs, shuffle=False, num_workers=self.hparams.nworkers)


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    from transformers import LayoutLMTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--layoutlm_str", type=str, default="microsoft/layoutlm-base-uncased")
    parser.add_argument("--t5_str", type=str, default="t5-base")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--bs", type=float, default=2)
    parser.add_argument("--train_transform", type=object, default=None)
    parser.add_argument("--eval_transform", type=object, default=None)
    parser.add_argument("--nworkers", type=object, default=mp.cpu_count())
    hparams = parser.parse_args()

    model = LayoutLMT5(hparams)

    simulated_input = LayoutLMTokenizer.from_pretrained(hparams.layoutlm_str).encode("Hello World.",
                                                                                     padding='max_length',
                                                                                     truncation=True,
                                                                                     max_length=hparams.seq_len,
                                                                                     return_tensors='pt')[0].unsqueeze(0)

    bbox = torch.tensor(np.random.randint(0, 1000, size=(1, 512, 4)))

    print(simulated_input.shape, bbox.shape)

    model({"input_tokens": simulated_input,
           "bbox": bbox,
           "target": simulated_input})
