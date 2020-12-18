'''
Abstracts DocVQA
'''
import os
import json
import torch
import imageio
import numpy as np
import random
from tqdm import tqdm
from transformers import PreTrainedTokenizer, T5Tokenizer
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class DocVQA(Dataset):
    @staticmethod
    def full(tokenizer: PreTrainedTokenizer,
             transform: object = None,
             seq_len: int = 512,
             no_image: bool = False):
        dataset = ConcatDataset([DocVQA(mode, tokenizer, transform=transform,
                                        seq_len=seq_len, no_image=no_image) for mode in ["train", "val", "test"]])
        dataset.__setattr__("tokenizer", tokenizer)
        return dataset

    def __init__(self,
                 mode: str,
                 tokenizer: PreTrainedTokenizer,
                 transform: object = None,
                 seq_len: int = 512,
                 no_image: bool = False):
        '''
        mode: one of train, val and test.
        tokenizer_string: input tokenizer string for LayoutLM
        transform: transforms to be applied to the document image if applicable.
        seq_len: maximum sequence len of encoded tokens.
        no_image: if True, don't load document images.

        returns:
            dict:
                document: transformed document image.
                input_tokens: tokenized text contained in the document.
                input_text: text contained in the document.
                bboxes: bounding boxes for each OCR detection in the document, on the format [tl_col, tl_row, br_col, br_row].
        '''
        super().__init__()
        assert mode in ["train", "val", "test"]
        with open(f"data/raw/{mode}/{mode}_v1.0.json", 'r') as data_json_file:
            self.data_json = json.load(data_json_file)

        self.folder = f"data/raw/{mode}"
        self.tokenizer = tokenizer
        self.transform = transform
        self.seq_len = seq_len
        self.mode = mode
        self.no_image = no_image

        print(f"{self.mode} DocVQA folder {self.folder} tokenizer {self.tokenizer} transform {self.transform} seq_len {self.seq_len} "
              f"no_image {self.no_image}")

    def __len__(self):
        return len(self.data_json["data"])

    def __getitem__(self, i: int):
        data = self.data_json["data"][i]

        if self.no_image:
            document = "NA"
        else:
            document = np.array(imageio.imread(os.path.join(self.folder, data["image"])))

        ID = data["ucsf_document_id"] + '_' + data["ucsf_document_page_no"]
        ocr_file = os.path.join(self.folder, "ocr_results", ID + ".json")
        with open(ocr_file, 'r') as ocr_file:
            ocr_info = json.load(ocr_file)

        lines = ocr_info['recognitionResults'][0]['lines']
        nlines = len(lines)

        bboxes = []
        input_text = 'question: ' + data["question"] + ' input: '
        for line in range(nlines):
            input_text += lines[line]['text'] + ' '
            bbox = lines[line]['boundingBox']
            bboxes.append(bbox[:2] + bbox[4:6])
        bboxes += [[0, 0, 0, 0]] * (self.seq_len - len(bboxes))
        assert len(bboxes) == self.seq_len

        bboxes = torch.tensor(bboxes)

        target_text = random.choice(data["answers"]) if self.mode == "train" else data.get("answers", ["NA"])[0]
        target = self.tokenizer.encode(target_text,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.seq_len,
                                       return_tensors='pt')[0]

        input_tokens = self.tokenizer.encode_plus(input_text,
                                                  padding='max_length',
                                                  truncation=True,
                                                  max_length=self.seq_len,
                                                  return_tensors='pt',
                                                  return_token_type_ids=True)

        if self.transform is not None:
            document = self.transform(document)

        return_dict = {"document": document,
                       "input_ids": input_tokens["input_ids"].squeeze(),
                       "token_type_ids": input_tokens["token_type_ids"].squeeze(),
                       "attention_mask": input_tokens["attention_mask"].squeeze(),
                       "input_text": input_text,
                       "bboxes": bboxes,
                       "target": target,
                       "target_text": target_text}

        return return_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("-ni", action="store_true")
    args = parser.parse_args()

    full_docvqa = DocVQA.full(T5Tokenizer.from_pretrained("t5-small"), no_image=args.ni)

    if args.task == "load_test":
        try:
            out_lens = []
            for doc in tqdm(DataLoader(full_docvqa, batch_size=1, shuffle=False, num_workers=12), leave=True, position=0):
                out_lens.append(torch.count_nonzero(doc["target"]))

            print(f"Mean output length: {np.array(out_lens).mean()}")
        except KeyboardInterrupt:
            pass
    elif args.task == "random_sample":
        dl = DataLoader(full_docvqa, batch_size=1, shuffle=True, num_workers=0)
        sample = next(iter(dl))
        for k, v in sample.items():
            if k != "document":
                print(f"{k}: {v}")
