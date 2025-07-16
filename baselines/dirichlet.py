import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from octopy.octopy.metrics.conflict.conflict_change_rate import get_degree_of_conflict
from baselines.utils import AvgTrustedLoss
from octopy.octopy.uncertainty.quantification import Dirichlet
from octopy.octopy.uncertainty.loss import EvidentialLoss


class DirichletModel(pl.LightningModule):
    def __init__(self, model, id, num_classes=42, dropout=0., lr=1e-3, annealing_step=50, activation="exp", clamp_max=10):
        super(DirichletModel, self).__init__()
        self.num_classes = num_classes
        self.model = model(num_classes=num_classes,
                           monte_carlo=False, dropout=dropout, dirichlet=True, activation=activation, clamp_max=clamp_max, id=id)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        # self.criterion = AvgTrustedLoss(num_views=3)
        self.criterion = EvidentialLoss(
            num_classes=num_classes, device=self.device, annealing_step=annealing_step)
        self.quantification = Dirichlet(num_classes)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None
        self.dc = None
        self.evidences_per_modality = None
        self.uncertainty_per_modality = None
        self.lr = lr

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch)
        self.log('train_loss', loss)
        acc = self.train_acc(output, target)
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def shared_step(self, batch, output_probs_tensor=False):
        image, audio, text, target = batch
        fused_output, output_per_modality = self((image, audio, text))

        # loss = self.criterion(temp, target, fused_output)
        loss = self.criterion(fused_output,
                              output_per_modality, target, self.current_epoch)
        evidences_per_modality = torch.stack(
            list(output_per_modality.values()))
        dc = get_degree_of_conflict(output_per_modality)

        if output_probs_tensor:
            return loss, fused_output, target, output_per_modality
        return loss, fused_output, target

    def validation_step(self, batch, batch_idx):
        return self.share_not_training_step(batch)

    def test_step(self, batch, batch_idx):
        return self.share_not_training_step(batch)

    def share_not_training_step(self, batch):
        loss, output, target, evidences_per_modality = self.shared_step(
            batch, True)
        self.test_acc(output, target)
        temps_per_modality = torch.stack(
            list(evidences_per_modality.values()))
        print(temps_per_modality.shape, "temps_per_modality from dirichlet")

        epistemic_uncertainty, aleatoric_uncertainty = self.quantification(
            output)

        epistemic_uncertainty_per_modality = []
        aleatoric_uncertainty_per_modality = []
        for e in temps_per_modality:
            epistemic_uncertainty, aleatoric_uncertainty = self.quantification(
                e)
            epistemic_uncertainty_per_modality.append(epistemic_uncertainty)
            aleatoric_uncertainty_per_modality.append(aleatoric_uncertainty)

        epistemic_uncertainty_per_modality = torch.stack(
            epistemic_uncertainty_per_modality)
        aleatoric_uncertainty_per_modality = torch.stack(
            aleatoric_uncertainty_per_modality)

        # Conflict base on dbf

        Y_pre = torch.argmax(output, dim=1)
        corrects = (Y_pre == target)
        num_correct = corrects.sum().item()
        num_sample = target.shape[0]
        num_classes = output.shape[1]
        uncertainty_dbf = num_classes / (output + 1).sum(dim=-1).unsqueeze(-1)
        print(uncertainty_dbf.shape, "uncertainty_dbf shape")

        # Uncertainty per modality based on dbf

        modality_uncertainties = []
        for e in evidences_per_modality:
            modality_uncertainties.append(
                num_classes / (evidences_per_modality[e] + 1).sum(dim=-1).unsqueeze(-1))
        modality_uncertainties = torch.stack(modality_uncertainties)

        return loss, output, target, epistemic_uncertainty, aleatoric_uncertainty, temps_per_modality, epistemic_uncertainty_per_modality, aleatoric_uncertainty_per_modality, uncertainty_dbf, num_sample, num_correct, modality_uncertainties

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        # self.criterion.annealing_step += 1

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_loss', np.mean(
            [x[0].detach().cpu().numpy() for x in outputs]), prog_bar=True)
        self.log('val_entropy', torch.cat(
            [x[3] for x in outputs]).mean(), prog_bar=True)
        self.log('val_sigma', torch.cat([x[4]
                 for x in outputs]).mean(), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_entropy_epi', torch.cat([x[3] for x in outputs]).mean())
        self.log('test_ale', torch.cat([x[4] for x in outputs]).mean())
        self.aleatoric_uncertainties = torch.cat(
            [x[4] for x in outputs]).detach().cpu().numpy()
        self.epistemic_uncertainties = torch.cat(
            [x[3] for x in outputs]).detach().cpu().numpy()
        self.evidences_per_modality = torch.cat(
            [x[5] for x in outputs], dim=1).detach().cpu().numpy()
        self.epistemic_uncertainty_per_modality = torch.cat(
            [x[6] for x in outputs], dim=1).detach().cpu().numpy()
        self.aleatoric_uncertainty_per_modality = torch.cat(
            [x[7] for x in outputs], dim=1).detach().cpu().numpy()
        self.uncertainty_dbf = torch.cat(
            [x[8] for x in outputs]).detach().cpu().numpy()

        num_samples = np.sum([x[9] for x in outputs])
        num_correct = np.sum([x[10] for x in outputs])
        self.modality_uncertainties = torch.cat(
            [x[11] for x in outputs], dim=1).detach().cpu().numpy()
        self.conflict_dbf = num_correct / num_samples
        print(self.conflict_dbf.shape, "conflict_dbf shape")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
