import argparse
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from torchaudio.transforms import MelSpectrogram
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Compose, Normalize

from baselines.classifiers import AudioClassifier, ImageClassifier, MultimodalClassifier, TextClassifier, MultimodalClassifierWithTwo
from baselines.unimodal_model import UnimodalModel
from baselines.de_model import DEModel
from baselines.dirichlet import DirichletModel
from baselines.mc_model import MCDModel
from data_generation.text_processing import extract_deep_text_features
from dataset import LUMADataset
from octopy.octopy.metrics.conflict.conflict_change_rate import get_degree_of_conflict
from lightning.pytorch.loggers import WandbLogger
import json


# Change Wandb name

pl.seed_everything(42)


class Text2FeatureTransform():
    def __init__(self, features_path):
        with open(features_path, 'rb') as f:
            self.features = np.load(f)

    def __call__(self, text, idx):
        return self.features[idx]


class PadCutToSizeAudioTransform():
    def __init__(self, size):
        self.size = size

    def __call__(self, audio):
        if audio.shape[-1] < self.size:
            audio = torch.nn.functional.pad(
                audio, (0, self.size - audio.shape[-1]))
        elif audio.shape[-1] > self.size:
            audio = audio[:, :self.size]
        return audio


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--noise_type', type=str, default='')
    parser.add_argument('-m', '--model', type=str, default='image')
    parser.add_argument('-a', '--annealing_step', type=int, default=50)
    parser.add_argument('-activation', '--activation', type=str, default="exp")
    parser.add_argument('-cm', '--clamp_max', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=150)
    parser.add_argument('-mo', '--mode', type=str, default="train")
    parser.add_argument('-i', '--id', type=str, default="")
    parser.add_argument('-fl', '--flambda', type=float, default=1)

    args, unknown = parser.parse_known_args()
    return args, unknown


args, unknown = parse_args()
if args.noise_type not in ['', 'diversity', 'label', 'sample']:
    raise ValueError('Invalid noise type')


train_audio_data_path = test_audio_data_path = ood_audio_data_path = 'data/audio' if args.noise_type != 'sample' else 'data/noisy_audio'
suffix = ''
if args.noise_type == 'diversity':
    suffix = '_diversity'
elif args.noise_type == 'label':
    suffix = '_noise_label'
elif args.noise_type == 'sample':
    suffix = '_noise'

train_audio_path = f'{train_audio_data_path}/datalist_train{suffix if args.noise_type != "sample" else ""}.csv'
test_audio_path = f'{test_audio_data_path}/datalist_test{suffix if args.noise_type not in ("sample", "diversity") else ""}.csv'
ood_audio_path = f'{ood_audio_data_path}/datalist_ood{suffix if args.noise_type != "sample" else ""}.csv'
train_image_path = f'data/image_data_train{suffix}.pickle'
train_text_path = f'data/text_data_train{suffix}.tsv'
test_image_path = f'data/image_data_test{suffix if args.noise_type != "diversity" else ""}.pickle'
test_text_path = f'data/text_data_test{suffix if args.noise_type != "diversity" else ""}.tsv'
ood_image_path = f'data/image_data_ood{suffix}.pickle'
ood_text_path = f'data/text_data_ood{suffix}.tsv'


# extract_deep_text_features(
#     train_text_path, output_path=f'text_features_train_{args.noise_type}.npy')
# extract_deep_text_features(
#     test_text_path, output_path=f'text_features_test_{args.noise_type}.npy')
# extract_deep_text_features(
#     ood_text_path, output_path=f'text_features_ood_{args.noise_type}.npy')

print(f'Loading data from {train_audio_path}, {test_audio_path}, {ood_audio_path}, {train_image_path}, {train_text_path}, {test_image_path}, {test_text_path}, {ood_image_path}, {ood_text_path}')
image_transform = Compose([
    ToTensor(),
    # Resize((224, 224)),
    Normalize(mean=(0.51, 0.49, 0.44),
              std=(0.27, 0.26, 0.28))
])
train_dataset = LUMADataset(train_image_path, train_audio_path, train_audio_data_path, train_text_path,
                            text_transform=Text2FeatureTransform(
                                f'text_features_train_{args.noise_type}.npy'),
                            audio_transform=Compose(
                                [MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                            image_transform=image_transform)

test_dataset = LUMADataset(test_image_path, test_audio_path, test_audio_data_path, test_text_path,
                           text_transform=Text2FeatureTransform(
                               f'text_features_test_{args.noise_type}.npy'),
                           audio_transform=Compose(
                               [MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                           image_transform=image_transform)


conflict_dataset = LUMADataset(test_image_path, test_audio_path, test_audio_data_path, test_text_path,
                               text_transform=Text2FeatureTransform(
                                   f'text_features_test_{args.noise_type}.npy'),
                               audio_transform=Compose(
                                   [MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                               image_transform=image_transform)

conflict_dataset.addConflict(1.0)

ood_dataset = LUMADataset(ood_image_path, ood_audio_path, ood_audio_data_path, ood_text_path,
                          text_transform=Text2FeatureTransform(
                              f'text_features_ood_{args.noise_type}.npy'),
                          audio_transform=Compose(
                              [MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                          image_transform=image_transform, ood=True)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)),
                                                                           len(train_dataset) - int(
                                                                               0.8 * len(train_dataset))])


batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
ood_loader = torch.utils.data.DataLoader(
    ood_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
conflict_loader = torch.utils.data.DataLoader(
    conflict_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Now we can use the loaders to train a model

acc_dict = {}
classes = 42
mc_samples = 100
dropout_p = 0.3
n_ensemble = 10

# mc_models = [MCDModel(c, classes, mc_samples, dropout_p) for c in [ImageClassifier, AudioClassifier, TextClassifier,
#                                                                    MultimodalClassifier]]
# de_models = [DEModel(c, classes, n_ensemble, dropout_p) for c in [ImageClassifier, AudioClassifier, TextClassifier,
#                                                                   MultimodalClassifier]]


if args.model == 'image':
    curr_model = ImageClassifier
elif args.model == 'audio':
    curr_model = AudioClassifier
elif args.model == 'text':
    curr_model = TextClassifier

base_name = f"{args.model}_{args.id}" if args.id != "" else f'model_{args.model}_a{args.annealing_step}_act{args.activation}_cm{args.clamp_max}_lr{args.learning_rate}_epochs{args.epochs}'

if args.model == 'multimodal':
    used_models = [DirichletModel(
        MultimodalClassifier, id=args.id, num_classes=classes, dropout=dropout_p, annealing_step=args.annealing_step,
        activation=args.activation, clamp_max=args.clamp_max, lr=args.learning_rate, flambda=args.flambda)]
elif args.model == "multimodalWithTwo":
    used_models = [DirichletModel(
        MultimodalClassifierWithTwo, id=args.id, num_classes=classes, dropout=dropout_p, annealing_step=args.annealing_step,
        activation=args.activation, clamp_max=args.clamp_max, lr=args.learning_rate, flambda=args.flambda)]
else:
    used_models = [UnimodalModel(curr_model, lr=args.learning_rate, annealing_step=args.annealing_step,
                                 activation=args.activation, clamp_max=args.clamp_max)]
# models = mc_models + de_models + dir_models
models = used_models

uncertainty_values = {}
dc_values = {}
wandb_logger = WandbLogger(
    log_model="all", name=base_name + f"_{args.mode}", project="LUMA_baselines")

print("Number of GPUS:", torch.cuda.device_count())
paths = [f"./unimodal_weights/{base_name}.pth"]
for classifier, path in zip(models, paths):
    model = classifier
    try:
        model_name = os.path.basename(path)
    except AttributeError:
        model_name = classifier.__class__.__name__ + \
            '_' + classifier.models[0].__class__.__name__

    max_epochs = args.epochs

    trainer = pl.Trainer(max_epochs=max_epochs,
                         gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                         callbacks=[pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_last=True)], logger=wandb_logger)

    if (args.mode == "train"):
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.model.state_dict(), path)
    if (args.mode == "test"):
        print(path)
        model.model.load_state_dict(torch.load(path, weights_only=True))
    print('Testing model')
    trainer.test(model, test_loader)
    acc_dict[model_name] = trainer.callback_metrics["test_acc"].item()
    if (args.model == 'multimodal' or args.model == "multimodalWithTwo"):
        acc_dict[model_name + '_ale'] = trainer.callback_metrics["test_ale"]
        acc_dict[model_name +
                 '_entropy_ep'] = trainer.callback_metrics["test_entropy_epi"]
        aleatoric_uncertainties = model.aleatoric_uncertainties
        epistemic_uncertainties = model.epistemic_uncertainties
    # if args.model == 'multimodal' or args.model == "multimodalWithTwo":
    #     print('Testing OOD')
    #     trainer.test(model, ood_loader)
    #     acc_dict[model_name + '_ood_ale'] = trainer.callback_metrics["test_ale"]
    #     acc_dict[model_name +
    #              '_ood'] = trainer.callback_metrics["test_acc"].item()
    #     acc_dict[model_name +
    #              '_ood_entropy_ep'] = trainer.callback_metrics["test_entropy_epi"]
    #     aleatoric_uncertainties_ood = model.aleatoric_uncertainties
    #     epistemic_uncertainties_ood = model.epistemic_uncertainties

    #     auc_score = roc_auc_score(
    #         np.concatenate([np.zeros(len(epistemic_uncertainties)),
    #                         np.ones(len(epistemic_uncertainties_ood))]),
    #         np.concatenate([epistemic_uncertainties, epistemic_uncertainties_ood]))

    if args.model == 'multimodal' or args.model == "multimodalWithTwo":
        uncertainty_values[f'epistemic'] = epistemic_uncertainties
        uncertainty_values[f'aleatoric'] = aleatoric_uncertainties
        # uncertainty_values["uncertainty_per_modality"] = model.uncertainty_per_modality
        uncertainty_values["evidences_per_modality"] = model.evidences_per_modality
        uncertainty_values["uncertainty_per_modality_dbf"] = model.modality_uncertainties
        uncertainty_values["uncertainty_dbf_normal"] = model.uncertainty_dbf
        uncertainty_values["conflict_dbf_normal"] = model.conflict_dbf

        dc_values[f'{model_name}_dc'] = model.dc
        # acc_dict[model_name + '_ood_auc'] = auc_score

        print("Testing conflict")

        trainer.test(model, conflict_loader)
        uncertainty_values["uncertainty_dbf_conflict"] = model.uncertainty_dbf


for key, value in acc_dict.items():
    print(f'{key}: {value}')

acc_df = pd.DataFrame.from_dict(acc_dict, orient="index")
acc_df.to_csv(f'./unimodal_results/{base_name}_f{args.flambda}.csv')

if args.model == 'multimodal' or args.model == "multimodalWithTwo":
    serializable_uncertainty_values = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in uncertainty_values.items()
    }

    with open(f"./unimodal_results/{base_name}_f{args.flambda}_uncertainty.json", 'w') as f:
        json.dump(serializable_uncertainty_values, f, indent=4)
