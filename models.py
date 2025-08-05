from utils import warmup_then_cosine_annealing_lr
from losses import LabelSmoothingLoss
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from lightning import LightningModule

import config
import torch
import pickle
import torch.nn.functional as F
import torchmetrics
import torchvision
import numpy as np
import typing


class CustomNetwork(LightningModule):
    """
    Classifier
    """

    def __init__(
        self,
        features_criterion: (
            nn.Module| None
        ),
        internal_features: int,
        is_pretrained: bool,
        model_name: str,
        warmup_epochs: int | None = None,
        unfreezed_layers: int | None = None,
        train_dl_size: int | None = None,
        max_epochs: int | None = None,
        base_lr: float | None = None,
        smoothing_ratio: float = 0,
        merged_loss_coef: float = 0,
        optim: torch.optim.Optimizer = torch.optim.Adam,
        **optim_kwargs
    ):
        super().__init__()
        assert merged_loss_coef >= 0
        self.merged_loss_coef = merged_loss_coef
        if self.merged_loss_coef > 0:
            assert features_criterion is not None
            self.aux_loss = LabelSmoothingLoss(smoothing=smoothing_ratio)
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.steps_per_epoch = train_dl_size
        self.loss_fn = features_criterion if features_criterion is not None else LabelSmoothingLoss(smoothing=smoothing_ratio)
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=config.CLASSES_CNT,
        )
        self.unfreezed_layers = unfreezed_layers
        self.optim = optim
        self.optim_kwargs = optim_kwargs
        self.warmup_epochs = warmup_epochs
        self.feature_criterion = features_criterion
        fetched_model = torchvision.models.get_model(model_name, weights = 'default' if is_pretrained else None)
        if self.unfreezed_layers is not None:
            self.pretrained_model = self._freeze_layers(fetched_model)
        self.pretrained_model.fc = nn.Linear(self.pretrained_model.fc.in_features, internal_features)
        self.pretrained_model = nn.Sequential(self.pretrained_model, 
                                              nn.BatchNorm1d(num_features=internal_features),
                                              nn.ReLU())
        self.tip = nn.ModuleDict(
            {
                'classification': nn.Sequential(
                    nn.Linear(internal_features, config.CLASSES_CNT),
                ),
                'extraction': nn.Sequential(
                    nn.Linear(internal_features, internal_features),
                    nn.BatchNorm1d(num_features=internal_features, affine=False))
            }
        )

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.base_lr, **self.optim_kwargs)
        total_steps = self.steps_per_epoch * self.max_epochs
        
        assert self.steps_per_epoch * self.warmup_epochs < total_steps

        lr_scheduler_config = {
            "scheduler" : warmup_then_cosine_annealing_lr(optimizer=optimizer,
                                                          start_factor=0.001,
                                                          total_steps=total_steps,
                                                          warmup_duration=self.steps_per_epoch * self.warmup_epochs),
            "interval" : "step",
            "frequency" : 1,
            "monitor": "val_loss",
            "strict": False,
            "name" : "warmup_with_cosannealing"
        }
        return [optimizer], [lr_scheduler_config]

    def _freeze_layers(self, pretrained_model):
        for child in list(pretrained_model.children())[:-self.unfreezed_layers]:
          for param in child.parameters():
              param.requires_grad = False
        return pretrained_model

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x = self.pretrained_model(x)
        if self.feature_criterion is not None:
            features = self.tip['extraction'](x)
            if self.merged_loss_coef > 0:
                return features, self.tip['classification'](x)
            else:
                return features
        else:
            return self.tip['classification'](x)

    def training_step(self, batch):
      return self._step(batch, "train")

    def validation_step(self, batch):
      return self._step(batch, "val")

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], kind: str) -> torch.Tensor:
      x, y = batch
      p = self(x)
      if self.merged_loss_coef > 0:
          features, probs = p
          loss = self.loss_fn(features, y) + self.merged_loss_coef * self.aux_loss(probs, y)
      else:
          loss = self.loss_fn(p, y)

      with torch.no_grad():
        if self.feature_criterion is None:
            accs = self.accuracy(p.argmax(axis=-1), y.argmax(axis=-1))
        else:
            accs = None
      return self._log_metrics(loss, accs, kind)

    def _log_metrics(self, loss, accs, kind):
        metrics = {}
        if loss is not None:
            metrics[f"{kind}_loss"] = loss.item()
        if accs is not None:
            metrics[f"{kind}_accs"] = accs

        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=kind=='train',
            on_epoch=True,
        )
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get("train_loss")
        print(f"Epoch {self.current_epoch} - Loss: {loss:.4f}")

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        pred = self.forward(x).cpu().numpy()
        return np.argmax(pred, axis=-1)


class ModelWithHead:
    def __init__(self, device: str, model: CustomNetwork, sklearn_model : ClassifierMixin | None = None) -> None:
        super().__init__()
        self.sklearn_model = sklearn_model
        self.model = model
        self.model.eval()
        self.device = torch.device(device)

        self.model.to(self.device)

    def get_model(self) -> CustomNetwork:
        return self.model

    def set_device(self, device: str) -> None:
        self.device = torch.device(device)

    def change_head(self, sklearn_model: ClassifierMixin) -> None:
        self.sklearn_model = sklearn_model

    def load_nn(self, nn_weights_path: str) -> None:
        checkpoint = torch.load(nn_weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)

    def load_head(self, knn_path: str) -> None:
        with open(knn_path, 'rb') as f:
            self.sklearn_model = pickle.load(f)

    def save_head(self, knn_path: str) -> None:
        with open(knn_path, 'wb') as f:
            pickle.dump(self.sklearn_model, f)

    @torch.no_grad()
    def train_head(self, indexloader: torch.utils.data.DataLoader) -> None:
        elems_per_class = dict()
        for x, y in indexloader:
            if isinstance(x, tuple):
                x = x[0] # Only fetch features vector
            assert isinstance(x, torch.Tensor)
            x = x.to(self.device)
            y = y.cpu().numpy()
            x = self.model.forward(x).cpu().numpy()
            x = x / np.linalg.norm(x, axis=-1, keepdims=True)
            cur_labels = np.argmax(y, axis=-1)
            for label in np.unique(cur_labels):
                if label in elems_per_class:
                    elems_per_class[label] = np.concatenate([elems_per_class[label], x[cur_labels==label]], axis=0)
                else:
                    elems_per_class[label] = x[cur_labels==label]
        labels = []
        embeds = []
        for label in elems_per_class:
            labels += [label] * elems_per_class[label].shape[0]
            embeds += [elems_per_class[label]]

        self.sklearn_model.fit(np.concatenate(embeds, axis=0), labels)

    @torch.no_grad()
    def extract_features(self, imgs: torch.Tensor):
        return self.model.forward(imgs).cpu().numpy()

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor | None, features: np.ndarray | None) -> np.ndarray:
        assert imgs is not None or features is not None

        if imgs is not None:
            imgs = imgs.to(self.device)

        if features is None:
            features = self.extract_features(imgs)

        features = features / np.linalg.norm(features, axis=-1).reshape((-1, 1))
        assert self.sklearn_model is not None
        pred = self.sklearn_model.predict(features)
        return pred
    

class CompositeModel:
    def __init__(
        self,
        classifier: CustomNetwork,
        model_with_head: ModelWithHead,
        device: str,
        rf: ClassifierMixin | None = None,
        knn: ClassifierMixin | None = None,
    ):
        super().__init__()

        self.classifer = classifier
        self.model_with_head = model_with_head
        self.model_with_head.set_device(device)
        self.device = torch.device(device)
        self.rf = rf
        self.knn = knn

        self.classifer.to(self.device)

    def load_models(self,
                    feature_extractor_ckpt_path: str | None,
                    classifier_ckpt_path: str | None,
                    rf_pickle : str | None,
                    knn_pickle: str | None) -> None:
        if classifier_ckpt_path is not None:
            classifier_ckpt = torch.load(classifier_ckpt_path, map_location=self.device)
            self.classifer.load_state_dict(classifier_ckpt['state_dict'])
            self.classifer.to(self.device)

        if feature_extractor_ckpt_path is not None:
            self.model_with_head.load_nn(feature_extractor_ckpt_path)

        if rf_pickle is not None:
            with open(rf_pickle, 'rb') as f:
                self.rf = pickle.load(f)
        
        if knn_pickle is not None:
            with open(knn_pickle, 'rb') as f:
                self.knn = pickle.load(f)

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor) -> np.ndarray:
        assert self.knn is not None and self.rf is not None

        imgs = imgs.to(self.device)

        features = self.model_with_head.extract_features(imgs)
        self.model_with_head.change_head(self.rf)
        type_preds = self.model_with_head.predict(imgs=None, features=features)
        rare_inds = (type_preds == 0).nonzero()[0]
        freq_inds = (type_preds == 1).nonzero()[0]

        preds = np.zeros(imgs.shape[0], dtype=np.int32)

        if rare_inds.size > 0:
            self.model_with_head.change_head(self.knn)
            rare_features = features[rare_inds]
            if rare_features.ndim < 2:
                rare_features = rare_features[..., None]
            cls_preds = self.model_with_head.predict(imgs=None, features=rare_features)
            preds[rare_inds] = cls_preds

        freq_imgs = imgs[torch.from_numpy(freq_inds).to(self.device)]
        cls_preds = self.classifer.predict(freq_imgs)
        preds[freq_inds] = cls_preds

        return preds