'''
Abstracts DocVQA
'''
import os
import json
import imageio
import glob
import multiprocessing as mp
from tqdm import tqdm
from transformers import LayoutLMTokenizer
from torch.utils.data import Dataset, DataLoader


def pre_load_worker(x):
    ID = os.path.basename(x[:-5])

    with open(x, 'r') as ocr_file:
        ocr_file_dict = json.load(ocr_file)

    return ID, ocr_file_dict


class DocVQA(Dataset):
    def __init__(self, mode: str, tokenizer_string: str = 'microsoft/layoutlm-base-uncased'):
        super().__init__()
        assert mode in ["train", "val", "test"]
        with open(f"data/raw/{mode}/{mode}_v1.0.json", 'r') as data_json_file:
            self.data_json = json.load(data_json_file)

        self.folder = f"data/raw/{mode}"

        # Pre-load all OCR results in memory.
        self.ocr = {}
        pool = mp.Pool()
        ocr_files = list(glob.glob(os.path.join(self.folder, "ocr_results", "*.json")))
        for ID, ocr_file_dict in tqdm(pool.imap_unordered(pre_load_worker, ocr_files), desc=f"Initializing {mode} DocVQA..."):
            self.ocr[ID] = ocr_file_dict

        self.tokenizer = LayoutLMTokenizer.from_pretrained(tokenizer_string)

    def __len__(self):
        return len(self.data_json["data"])

    def __getitem__(self, i: int):
        data = self.data_json["data"][i]
        ID = data["ucsf_document_id"] + '_' + data["ucsf_document_page_no"]
        document = imageio.imread(os.path.join(self.folder, data["image"]))
        ocr_info = self.ocr[ID]
        lines = ocr_info['recognitionResults'][0]['lines']
        nlines = len(lines)

        bboxes = []
        original_text = ''
        for line in range(nlines):
            original_text += lines[line]['text'] + ' '
            bbox = lines[line]['boundingBox']
            bboxes.append(bbox[:2] + bbox[4:6])

        return document, original_text, bboxes

    def get_dataloader(self, batch_size: int, shuffle: bool):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


if __name__ == "__main__":
    doc_vqa = DocVQA("train")
    print(doc_vqa[0][1], doc_vqa[0][2])
