import json
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import DonutProcessor

added_tokens = []


class DonutDataset(Dataset):
    """
    PyTorch Dataset for Donut.

    """

    def __init__(
        self,
        images_path: str = "/data/ubuntu/resume_skills/data/train/images_train",
        labels_path: str = "/data/ubuntu/resume_skills/data/train/labels",
        ignore_id: int = -100,
    ):
        super().__init__()

        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        self.images_paths = [p for p in Path(images_path).rglob("*")]
        self.labels_paths = [p for p in Path(labels_path).rglob("*")]
        self.file_name2label_path = {p.stem: p for p in self.labels_paths}
        self.ignore_id = ignore_id

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        image_path = self.images_paths[idx]
        label_path = self.file_name2label_path[image_path.stem]

        label_text = str(json.load(open(label_path)))

        # inputs
        image = Image.open(image_path)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        input_ids = self.processor.tokenizer(
            label_text,
            return_tensors="pt",
        )[
            "input_ids"
        ].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = (
            self.ignore_id
        )  # model doesn't need to predict pad token
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)

        return {"pixel_values": pixel_values, "labels": labels, "target_sequence": label_text}
