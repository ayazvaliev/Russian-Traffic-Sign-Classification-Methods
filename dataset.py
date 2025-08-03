import torch
import typing
import os
import json
import numpy as np

from utils import read_annotations_csv
from PIL import Image
import config
import transforms


class DatasetRTSD(torch.utils.data.Dataset):
    """Training Dataset"""

    def __init__(
        self,
        root_folders: typing.List[str],
        path_to_classes_json: str,
        for_training=False,
        type_classification=False,
        use_transform=True
    ) -> None:
        super().__init__()
        self.classes, self.class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
        self.idx_to_type = {self.class_to_idx[cls_name]: type_ for cls_name, type_ in DatasetRTSD.get_types(path_to_classes_json).items()}
        self.type_classification = type_classification
        self.samples = list()
        for dir_path in root_folders:
            for class_name in os.listdir(dir_path):
                for img_name in os.listdir(dir_path + '/' + class_name):
                    self.samples.append((dir_path + '/' + class_name + '/' + img_name, self.class_to_idx[class_name]))

        self.classes_to_samples = {i : [] for i in range(len(self.classes))}
        for i in range(len(self.samples)):
            class_id = self.samples[i][1]
            self.classes_to_samples[class_id].append(i)

        self.transform = transforms.DEFAULT_TRANSFORM_TRAIN if use_transform else transforms.DEFAULT_TRANSFORM_VALID
        self.for_training = for_training

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        img_path = self.samples[index][0]
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.type_classification:
            label = self.idx_to_type[self.samples[index][1]]
            label = 0 if label == 'rare' else 1
            one_hot_label = torch.zeros((2,))
            one_hot_label[label] = 1
        else:
            label = self.samples[index][1]
            one_hot_label = torch.zeros((config.CLASSES_CNT,))
            one_hot_label[label] = 1

        transformed_image = self.transform(image=img)["image"]
        if self.for_training:
            return transformed_image, one_hot_label
        else:
            return transformed_image, self.samples[index][0], label

    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        with open(path_to_classes_json, 'r') as file:
            data = json.load(file)
        
        classes = list()
        class_to_idx = dict()
        for class_name in data.keys():
            class_to_idx[class_name] = data[class_name]['id']
            classes.append(class_name)
        return classes, class_to_idx
    
    @staticmethod
    def get_types(path_to_classes_json):
        with open(path_to_classes_json, 'r') as file:
            data = json.load(file)
        
        types = dict()
        for class_name in data.keys():
            types[class_name] = data[class_name]['type']
        return types

    def __len__(self) -> int:
        return len(self.samples)


class TestData(torch.utils.data.Dataset):
    """
    Test dataset with rare signs
    """

    def __init__(
        self,
        root: str,
        path_to_classes_json: str,
        annotations_file: str = None,
        for_validating = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.classes, self.class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
        self.samples = list()
        for img_name in os.listdir(root):
            self.samples.append(img_name)

        self.transform = transforms.DEFAULT_TRANSFORM_VALID

        self.targets = None
        if annotations_file is not None:
            self.targets = read_annotations_csv(annotations_file)
            for filename in self.targets:
                self.targets[filename] = self.class_to_idx[self.targets[filename]]
        self.is_validating = for_validating


    def __getitem__(self, index: int | list[int]) -> typing.Tuple[torch.Tensor, str, int]:
        if type(index) is int:
            index_list = [index]
        else:
            index_list = index

        stacked = []
        for index in index_list:
            img_path = self.root + '/' + self.samples[index]
            img = np.array(Image.open(img_path).convert("RGB"))
            transformed_image = self.transform(image=img)["image"]

            if self.is_validating:
                assert self.targets is not None

                label = self.targets[self.samples[index]]
                one_hot_label = torch.zeros((config.CLASSES_CNT,))
                one_hot_label[label] = 1
                stacked.append((transformed_image, one_hot_label))
            else:
                stacked.append((transformed_image, self.samples[index], -1 if self.targets is None else self.targets[self.samples[index]]))
        return stacked[0] if len(stacked) == 1 else stacked
    def __len__(self) -> int:
        return len(self.samples)
