import torch
import typing
import json
import config
from utils import read_annotations_csv, calc_metric
from dataset import TestData


def apply_classifier(
    model: torch.nn.Module,
    test_folder: str,
    path_to_classes_json: str,
) -> typing.List[typing.Mapping[str, typing.Any]]:
    batch_size = config.BATCH_SIZE
    test_dataset = TestData(root=test_folder, 
                            path_to_classes_json=path_to_classes_json,
                            for_validating=False)
    results = []
    for i in range(0, len(test_dataset), batch_size):
        tensor_batch = []
        name_batch = []
        for j in range(i, min(i + batch_size, len(test_dataset))):
            img_tensor, img_name, _ = test_dataset[j]
            tensor_batch.append(img_tensor)
            name_batch.append(img_name)
        tensor_batch = torch.stack(tensor_batch).to(model.device)
        preds = model.predict(tensor_batch)

        N = len(name_batch)
        for i in range(N):
            entry = {'filename': name_batch[i], 'class': test_dataset.classes[preds[i]]}
            results.append(entry)
    return results


def test_classifier(model, test_folder, annotations_file, classes_info_path):
    gt = read_annotations_csv(annotations_file)
    y_pred = []
    y_true = []
    output = apply_classifier(model=model,
                                test_folder=test_folder,
                                path_to_classes_json=classes_info_path)
    output = {elem["filename"]: elem["class"] for elem in output}
    for k, v in output.items():
        y_pred.append(v)
        y_true.append(gt[k])

    with open(classes_info_path, "r") as fr:
        classes_info = json.load(fr)
        class_name_to_type = {k: v["type"] for k, v in classes_info.items()}

        total_tuple = calc_metric(y_true, y_pred, "all", class_name_to_type)
        rare_tuple = calc_metric(y_true, y_pred, "rare", class_name_to_type)
        freq_tuple = calc_metric(y_true, y_pred, "freq", class_name_to_type)
    return total_tuple, rare_tuple, freq_tuple

