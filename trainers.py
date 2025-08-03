import torch
import os
import config

from lightning import Trainer
from pathlib import Path
from models import CustomNetwork, ModelWithHead, CompositeModel
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import DatasetRTSD, TestData
from samplers import CustomBatchSampler, IndexSampler
from losses import FeaturesLoss
from config import TRAIN_DATASET_PATH, SYNT_DATASET_PATH, TEST_DATASET_PATH, CLASSES_INFO_PATH, TEST_ANNOTATIONS_PATH, CKPTS_PATH, DEVICE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

__all__ = [
    'train_simple_classifier',
    'train_classifier_with_synt',
    'train_feature_extractor_knn_pipeline',
    'train_composite_model_pipeline'
]

def train_simple_classifier() -> torch.nn.Module:
    """
    Fine-tunes pretrained resnet-50 classifer with frequent signs
    """

    train_dataset = DatasetRTSD(root_folders=[TRAIN_DATASET_PATH], 
                                path_to_classes_json=CLASSES_INFO_PATH, 
                                for_training=True)

    test_dataset = TestData(root=TEST_DATASET_PATH, 
                            path_to_classes_json=CLASSES_INFO_PATH, 
                            annotations_file=TEST_ANNOTATIONS_PATH,
                            for_validating=True)

    dl_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count()
    )

    dl_valid = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=os.cpu_count()
    )

    if CKPTS_PATH is not None:
        checkpoint_callback = [ModelCheckpoint(
            dirpath=CKPTS_PATH,
            filename="baseline_classifier-{epoch}-{val_accs:.2f}",
            monitor="val_accs",
            mode="max",
            save_last=True
        )]
    else:
        checkpoint_callback = None
    
    trainer = Trainer(accelerator=DEVICE,
                        max_epochs=config.MAX_EPOCHS,
                        enable_checkpointing=CKPTS_PATH is not None,
                        logger=False,
                        callbacks=checkpoint_callback)

    model = CustomNetwork(
        features_criterion=None,
        internal_features=config.INTERNAL_FEATURES,
        is_pretrained=True,
        train_dl_size=len(dl_train),
        max_epochs=trainer.max_epochs,
        base_lr=config.BASE_LR,
        unfreezed_layers=config.UNFREEZED_LAYERS,
        warmup_epochs=config.WARMUP_EPOCHS,
        model_name=config.NET_NAME,
        smoothing_ratio=config.SMOOTHING_RATIO,
        optim=torch.optim.Adam
    )
    trainer.fit(model, dl_train, dl_valid)

    return model


def train_classifier_with_synt() -> torch.nn.Module:
    """
    Fine-tunes pretrained resnet-50 classifer with frequent and syntetic rare signs
    """  

    train_dataset = DatasetRTSD(root_folders=[TRAIN_DATASET_PATH, 
                                              SYNT_DATASET_PATH], 
                                path_to_classes_json=CLASSES_INFO_PATH, 
                                for_training=True)

    test_dataset = TestData(root=TEST_DATASET_PATH, 
                            path_to_classes_json=CLASSES_INFO_PATH, 
                            annotations_file=TEST_ANNOTATIONS_PATH,
                            for_validating=True)

    dl_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count()
    )
    dl_valid = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=os.cpu_count()
    )

    if CKPTS_PATH is not None:
        checkpoint_callback = [ModelCheckpoint(
            dirpath=CKPTS_PATH,
            filename="synt_classifier-{epoch}-{val_accs:.2f}",
            monitor="val_accs",
            mode="max",
            save_last=True
        )]
    else:
        checkpoint_callback = None
    
    trainer = Trainer(accelerator=DEVICE,
                        max_epochs=config.MAX_EPOCHS,
                        enable_checkpointing=CKPTS_PATH is not None,
                        logger=False,
                        callbacks=checkpoint_callback)

    model = CustomNetwork(
        features_criterion=None,
        internal_features=config.INTERNAL_FEATURES,
        is_pretrained=True,
        train_dl_size=len(dl_train),
        max_epochs=trainer.max_epochs,
        base_lr=config.BASE_LR,
        unfreezed_layers=config.UNFREEZED_LAYERS,
        warmup_epochs=config.WARMUP_EPOCHS,
        model_name=config.NET_NAME,
        smoothing_ratio=config.SMOOTHING_RATIO,
        optim=torch.optim.Adam
    )
    trainer.fit(model, dl_train, dl_valid)

    return model


def train_feature_extractor() -> torch.nn.Module:
    """
    Trains feature extraction model on contrastive loss task
    """

    train_dataset = DatasetRTSD(root_folders=[TRAIN_DATASET_PATH, SYNT_DATASET_PATH],
                                path_to_classes_json=CLASSES_INFO_PATH,
                                for_training=True)
    elems_per_class = config.ELEMS_PER_CLASS
    classes_per_batch = config.CLASSES_PER_BATCH
    batch_size = elems_per_class * classes_per_batch
    sampler=CustomBatchSampler(train_dataset, elems_per_class=elems_per_class, classes_per_batch=classes_per_batch)

    dl_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=os.cpu_count(),
        sampler=sampler
    )

    if CKPTS_PATH is not None:
        checkpoint_callback = [ModelCheckpoint(
            dirpath=CKPTS_PATH,
            filename=str(f"feature_extractor-{{epoch}}-{{train_loss:.2f}}-{config.MERGED_LOSS_COEF:.2f}"),
            monitor="train_loss",
            mode="min",
            save_last=True
        )]
    else:
        checkpoint_callback = None

    trainer = Trainer(callbacks=checkpoint_callback, 
                        accelerator=DEVICE,
                        max_epochs=config.MAX_EPOCHS,
                        enable_checkpointing=CKPTS_PATH is not None,
                        logger=False)

    model = CustomNetwork(
        features_criterion=FeaturesLoss(margin=config.MARGIN),
        internal_features=config.INTERNAL_FEATURES,
        is_pretrained=True,
        train_dl_size=len(dl_train),
        max_epochs=trainer.max_epochs,
        base_lr=config.BASE_LR,
        merged_loss_coef=config.MERGED_LOSS_COEF,
        unfreezed_layers=config.UNFREEZED_LAYERS,
        warmup_epochs=config.WARMUP_EPOCHS,
        model_name=config.NET_NAME,
        smoothing_ratio=config.SMOOTHING_RATIO,
        optim=torch.optim.Adam
    )
    trainer.fit(model, train_dataloaders=dl_train)

    return model


def train_head(head_save_name: str,
               model_path: str,
               sklearn_model_class,
               examples_per_class,
               model_prefix='model_',
               nn_prefix='nn_',
               dataset_prefix='dataset_',
               **kwargs) -> ModelWithHead:
    model_kwargs = {k.removeprefix(model_prefix) : v for k, v in kwargs.items() if k.startswith(model_prefix)}
    nn_kwargs = {k.removeprefix(nn_prefix) : v for k, v in kwargs.items() if k.startswith(nn_prefix)}
    dataset_kwargs = {k.removeprefix(dataset_prefix) : v for k, v in kwargs.items() if k.startswith(dataset_prefix)}

    nn = CustomNetwork(**nn_kwargs)
    sklearn_model = sklearn_model_class(**model_kwargs)
    model = ModelWithHead(sklearn_model=sklearn_model, model=nn, device=DEVICE, **model_kwargs)
    model.load_nn(model_path)

    root_folders = [SYNT_DATASET_PATH]
    if not config.SYNT_ONLY:
        root_folders.append(TRAIN_DATASET_PATH)
    train_dataset = DatasetRTSD(root_folders=root_folders,
                                path_to_classes_json=CLASSES_INFO_PATH,
                                for_training=True,
                                **dataset_kwargs)

    sampler = IndexSampler(data_source=train_dataset, examples_per_class=examples_per_class)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=os.cpu_count(),
        batch_size=config.BATCH_SIZE,
        drop_last=False,
        sampler=sampler
    )
    model.train_head(train_loader)
    model.save_head(Path(CKPTS_PATH) / Path(head_save_name))

    return model


def train_feature_extractor_knn_pipeline() -> ModelWithHead:
    train_feature_extractor()
    model_path = None
    for filename in os.listdir(CKPTS_PATH):
        if filename.startswith('feature_extractor-'):
            model_path = Path(CKPTS_PATH) / Path(filename)
            break
    assert model_path is not None

    return train_head(
        head_save_name=f'knn-n_neighbors={config.N_NEIGHBORS}-index_per_class={config.INDEX_PER_CLASS}-synt_only-{config.SYNT_ONLY}.bin',
        model_path=model_path,
        sklearn_model_class=KNeighborsClassifier,
        examples_per_class=config.INDEX_PER_CLASS,
        model_n_neighbors=config.N_NEIGHBORS,
        nn_is_pretrained=False,
        nn_features_criterion=FeaturesLoss(margin=config.MARGIN),
        dataset_use_transform=False
    )


def train_composite_model_pipeline() -> CompositeModel:
    feature_extractor = train_feature_extractor()
    classifier = train_classifier_with_synt()

    extractor_path = None
    classifier_path = None
    for filename in os.listdir(CKPTS_PATH):
        if filename.startswith('feature_extractor-'):
            extractor_path = Path(CKPTS_PATH) / Path(filename)
        if filename.startswith('synt_classifier-'):
            classifier_path = Path(CKPTS_PATH) / Path(filename)
    assert extractor_path and classifier_path

    knn_name = Path(f'knn_composite-n_neighbors={config.N_NEIGHBORS}-index_per_class={config.INDEX_PER_CLASS}.bin')
    rf_name = Path(f'rf_composite-n_estimators={config.N_ESTIMATORS}.bin')

    knn_model = train_head(
        head_save_name=knn_name,
        model_path=extractor_path,
        sklearn_model_class=KNeighborsClassifier,
        examples_per_class=config.INDEX_PER_CLASS,
        model_n_neighbors=config.N_NEIGHBORS,
        nn_is_pretrained=False,
        nn_features_criterion=FeaturesLoss(margin=config.MARGIN),
        dataset_use_transform=False
    )

    train_head(
        head_save_name=rf_name,
        model_path=extractor_path,
        sklearn_model_class=RandomForestClassifier,
        examples_per_class=None,
        model_n_estimators = config.N_ESTIMATORS,
        model_n_jobs=-1,
        nn_is_pretrained=False,
        nn_features_criterion=FeaturesLoss(margin=config.MARGIN),
        dataset_use_transform=False,
        dataset_type_classification=True,
    )

    classifier = CustomNetwork(
        internal_features=config.INTERNAL_FEATURES,
        is_pretrained=False,
        model_name=config.NET_NAME
    )

    composite_model = CompositeModel(classifier=classifier,
                                     model_with_head=feature_extractor,
                                     device=DEVICE)
    composite_model.load_models(
        feature_extractor_ckpt=extractor_path,
        classifier_ckpt=classifier_path,
        knn_pickle=Path(CKPTS_PATH) / knn_name,
        rf_pickle=Path(CKPTS_PATH) / rf_name
    )

    return composite_model