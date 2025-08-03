import argparse
import yaml

from utils import update_global_vars, format_test_results
from evaluate import test_classifier
from transforms import init_transforms
from trainers import *
from config import TEST_DATASET_PATH, TEST_ANNOTATIONS_PATH, CLASSES_INFO_PATH

parser = argparse.ArgumentParser(description="Trains classification model based on selected config")
parser.add_argument("--config", type=str, help="path to YAML config file", required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as f_yaml:
        config_data = yaml.safe_load(f_yaml)
    
    update_global_vars(config_data)
    init_transforms()

    if config_data['model_name'] == 'softmax_classifier':
        model = train_simple_classifier()
    elif config_data['model_name'] == 'softmax_classifier_with_synt':
        model = train_classifier_with_synt()
    elif config_data['model_name'] == 'feature_extractor+knn':
        model = train_feature_extractor_knn_pipeline()
    elif config_data['model_name'] == 'composite':
        model = train_composite_model_pipeline()
    else:
        raise NameError()
    
    print(f'Eval results for model_name={config_data['model_name']}')
    format_test_results(
        *test_classifier(
            model=model,
            test_folder=TEST_DATASET_PATH,
            annotations_file=TEST_ANNOTATIONS_PATH,
            classes_info_path=CLASSES_INFO_PATH
        )
    )