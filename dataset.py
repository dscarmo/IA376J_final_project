'''
Abstracts DocVQA
'''
import os
import json
import torch
import imageio
import numpy as np
from tqdm import tqdm
from transformers import LayoutLMTokenizer
from torch.utils.data import Dataset, ConcatDataset, DataLoader


def pre_load_worker(x):
    ID = os.path.basename(x[:-5])

    with open(x, 'r') as ocr_file:
        ocr_file_dict = json.load(ocr_file)

    return ID, ocr_file_dict


class DocVQA(Dataset):
    @staticmethod
    def full(tokenizer_string: str = 'microsoft/layoutlm-base-uncased',
             transform=None,
             seq_len=512):
        dataset = ConcatDataset([DocVQA(mode, tokenizer_string=tokenizer_string, transform=transform,
                                        seq_len=seq_len) for mode in ["train", "val", "test"]])
        dataset.__setattr__("tokenizer", dataset.datasets[0].tokenizer)
        return dataset

    def __init__(self,
                 mode: str,
                 tokenizer_string: str = 'microsoft/layoutlm-base-uncased',
                 transform=None,
                 seq_len=512):
        super().__init__()
        assert mode in ["train", "val", "test"]
        with open(f"data/raw/{mode}/{mode}_v1.0.json", 'r') as data_json_file:
            self.data_json = json.load(data_json_file)

        self.folder = f"data/raw/{mode}"
        self.tokenizer = LayoutLMTokenizer.from_pretrained(tokenizer_string)
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_json["data"])

    def __getitem__(self, i: int):
        data = self.data_json["data"][i]
        document = np.array(imageio.imread(os.path.join(self.folder, data["image"])))

        ID = data["ucsf_document_id"] + '_' + data["ucsf_document_page_no"]
        ocr_file = os.path.join(self.folder, "ocr_results", ID + ".json")
        with open(ocr_file, 'r') as ocr_file:
            ocr_info = json.load(ocr_file)

        lines = ocr_info['recognitionResults'][0]['lines']
        nlines = len(lines)

        bboxes = []
        original_text = ''
        for line in range(nlines):
            original_text += lines[line]['text'] + ' '
            bbox = lines[line]['boundingBox']
            bboxes.append(bbox[:2] + bbox[4:6])

        tokens = self.tokenizer.encode(original_text,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.seq_len,
                                       return_tensors='pt')[0]

        bboxes = torch.tensor(bboxes)

        if self.transform is not None:
            document = self.transform(document)

        return {"document": document, "tokens": tokens, "bboxes": bboxes, "original_text": original_text}


if __name__ == "__main__":
    full_docvqa = DocVQA.full()
    shapes = []
    text_lens = []

    for doc in tqdm(DataLoader(full_docvqa, batch_size=1, shuffle=False, num_workers=12), leave=True, position=0):
        shapes.append(doc["document"][0].shape)
        text_lens.append(len(doc["original_text"][0].split()))

    print(np.array(shapes).mean(axis=0))
    print(np.array(text_lens).mean())
