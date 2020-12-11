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
from transformers import LayoutLMTokenizer
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class DocVQA(Dataset):
    @staticmethod
    def full(tokenizer_string: str = 'microsoft/layoutlm-base-uncased',
             transform=None,
             seq_len=512,
             no_image=False):
        dataset = ConcatDataset([DocVQA(mode, tokenizer_string=tokenizer_string, transform=transform,
                                        seq_len=seq_len, no_image=no_image) for mode in ["train", "val", "test"]])
        dataset.__setattr__("tokenizer", dataset.datasets[0].tokenizer)
        return dataset

    def __init__(self,
                 mode: str,
                 tokenizer_string: str = 'microsoft/layoutlm-base-uncased',
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
        self.tokenizer = LayoutLMTokenizer.from_pretrained(tokenizer_string)
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
        input_text = ''
        for line in range(nlines):
            input_text += lines[line]['text'] + ' '
            bbox = lines[line]['boundingBox']
            bboxes.append(bbox[:2] + bbox[4:6])

        input_tokens = self.tokenizer.encode(input_text,
                                             padding='max_length',
                                             truncation=True,
                                             max_length=self.seq_len,
                                             return_tensors='pt')[0]

        bboxes = torch.tensor(bboxes)

        target_text = random.choice(data["answers"]) if self.mode == "train" else data.get("answers", ["NA"])[0]
        target = self.tokenizer.encode(target_text,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.seq_len,
                                       return_tensors='pt')[0]

        if self.transform is not None:
            document = self.transform(document)

        return {"document": document,
                "input_tokens": input_tokens, "input_text": input_text,
                "bboxes": bboxes,
                "target": target, "target_text": target_text}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ni", action="store_true")
    args = parser.parse_args()

    full_docvqa = DocVQA.full(no_image=args.ni)

    try:
        for doc in tqdm(DataLoader(full_docvqa, batch_size=1, shuffle=False, num_workers=12), leave=True, position=0):
            pass
    except KeyboardInterrupt:
        pass
