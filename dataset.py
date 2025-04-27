import os
import glob
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from torchvision import transforms
from PIL import Image
from config import tag2idx, idx2tag, max_node, log_fre

segment_entities = ["person", "location", "miscellaneous", "organization"]
entity2cropname = {
    "person": "crop_person",
    "location": "crop_location",
    "miscellaneous": "crop_miscellaneous",
    "organization": "crop_organization"
}

class MMNerDataset(Dataset):
    def __init__(self, my_data_dir, text_segments_dir, imgdir="./data/twitter2015/image", split="train", max_length=128):
        self.split_dir = os.path.join(my_data_dir, split)
        self.text_segments_dir = text_segments_dir
        self.imgdir = imgdir
        self.max_length = max_length

        self.s_files = sorted(glob.glob(os.path.join(self.split_dir, "*_s.txt")))
        self.l_files = sorted(glob.glob(os.path.join(self.split_dir, "*_l.txt")))
        self.p_files = sorted(glob.glob(os.path.join(self.split_dir, "*_p.txt")))

        self.tokenizer = BertTokenizerFast.from_pretrained("./bert-base-cased")

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.s_files)

    def __getitem__(self, idx):
        with open(self.s_files[idx], "r", encoding="utf-8") as f:
            second_text = f.readline().strip()
        with open(self.l_files[idx], "r", encoding="utf-8") as f:
            gold_labels = f.readline().strip().split("\t")
        with open(self.p_files[idx], "r", encoding="utf-8") as f:
            seg_digit = f.readline().strip()

        seg_path = os.path.join(self.text_segments_dir, seg_digit + ".txt")
        with open(seg_path, "r", encoding="utf-8") as f:
            segments = [line.strip() for line in f if line.strip()]
            segments = segments[:4]
            text_segments = "\n".join(segments)

        # Tokenization + 标签
        tokens = ["[CLS]"]
        label_ids = [tag2idx["CLS"]]
        for word, label in zip(second_text.split(), gold_labels):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            for i, _ in enumerate(word_tokens):
                label_ids.append(tag2idx[label] if i == 0 else tag2idx["X"])
        tokens = tokens[:self.max_length - 1]
        tokens.append("[SEP]")
        label_ids = label_ids[:self.max_length - 1]
        label_ids.append(tag2idx["SEP"])

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        input_ids.extend([0] * pad_len)
        mask.extend([0] * pad_len)
        label_ids.extend([tag2idx["PAD"]] * pad_len)

        # 图像加载，顺序与 segments 顺序对应
        imgid = seg_digit  # 图像 ID 与 segment 编号一致
        img_tensors = []
        for entity in segment_entities:
            cropname = entity2cropname[entity]
            img_path = os.path.join(self.imgdir, f"{imgid}/{cropname}_{imgid}.jpg")
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = self.preprocess(image)
            except Exception as e:
                print(f"加载图像失败: {img_path} -> {e}")
                image_tensor = torch.zeros(3, 224, 224)
            img_tensors.append(image_tensor)
        img_tensor_stack = torch.stack(img_tensors)  # (4, 3, 224, 224)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "second_text": second_text,
            "gold_labels": gold_labels,
            "text_segments": text_segments,
            "image_tensor": img_tensor_stack  # shape: (4, 3, 224, 224)
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    label_ids = torch.stack([item["label_ids"] for item in batch])
    image_tensor = torch.stack([item["image_tensor"] for item in batch])  # shape: (B, 4, 3, 224, 224)

    second_texts = [item["second_text"] for item in batch]
    gold_labels = [item["gold_labels"] for item in batch]
    text_segments = [item["text_segments"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "image_tensor": image_tensor,
        "second_text": second_texts,
        "gold_labels": gold_labels,
        "text_segments": text_segments,
    }
