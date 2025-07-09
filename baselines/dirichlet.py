import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from octopy.octopy.metrics.conflict.conflict_change_rate import get_degree_of_conflict
from baselines.utils import AvgTrustedLoss
from octopy.octopy.uncertainty.quantification import Dirichlet
from octopy.octopy.uncertainty.loss import EvidentialLoss


class DirichletModel(pl.LightningModule):
    def __init__(self, model, num_classes=42, dropout=0.):
        super(DirichletModel, self).__init__()
        self.num_classes = num_classes
        self.model = model(num_classes=num_classes,
                           monte_carlo=False, dropout=dropout, dirichlet=True)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.criterion = AvgTrustedLoss(num_views=3)
        # self.criterion = EvidentialLoss(
        #     num_classes=num_classes, device=self.device)
        self.quantification = Dirichlet(num_classes)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None
        self.dc = None
        self.evidences_per_modality = None
        self.uncertainty_per_modality = None

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
        temp = torch.stack(
            list(output_per_modality.values()))
        loss = self.criterion(temp, target, fused_output)
        # loss = self.criterion(fused_output,
        #                       output_per_modality, target, self.current_epoch)
        if output_probs_tensor:
            return loss, fused_output, target, output_per_modality
        return loss, fused_output, target

    def validation_step(self, batch, batch_idx):
        loss, output, target, evidences_per_modality = self.shared_step(
            batch, True)
        evidences_per_modality = torch.stack(
            list(evidences_per_modality.values()))
        self.val_acc(output, target)
        alphas = output + 1
        probs = alphas / alphas.sum(dim=-1, keepdim=True)
        epistemic_uncertainty = self.num_classes / alphas.sum(dim=-1)
        alpha_0 = alphas.sum(dim=-1, keepdim=True)
        aleatoric_uncertainty = - \
            torch.sum(probs * (torch.digamma(alphas + 1) -
                      torch.digamma(alpha_0 + 1)), dim=-1)

        uncertainty_for_dc = self.num_classes / \
            (evidences_per_modality + 1).sum(dim=-1).unsqueeze(-1)
        epistemic_uncertainty_per_modality = torch.stack([self.quantification(
            e)[0] for e in evidences_per_modality])
        aleatoric_uncertainty_per_modality = torch.stack([self.quantification(
            e)[0] for e in evidences_per_modality])
        # dc = get_degree_of_conflict(evidences_per_modality, uncertainty_for_dc)
        return loss, output, target, epistemic_uncertainty, aleatoric_uncertainty, aleatoric_uncertainty, evidences_per_modality, epistemic_uncertainty_per_modality

    def test_step(self, batch, batch_idx):
        loss, output, target, evidences_per_modality = self.shared_step(
            batch, True)
        self.test_acc(output, target)
        evidences_per_modality = torch.stack(
            list(evidences_per_modality.values()))
        alphas = output + 1
        probs = alphas / alphas.sum(dim=-1, keepdim=True)
        epistemic_uncertainty = self.num_classes / alphas.sum(dim=-1)
        alpha_0 = alphas.sum(dim=-1, keepdim=True)
        aleatoric_uncertainty = - \
            torch.sum(probs * (torch.digamma(alphas + 1) -
                      torch.digamma(alpha_0 + 1)), dim=-1)

        uncertainty_for_dc = self.num_classes / \
            (evidences_per_modality + 1).sum(dim=-1).unsqueeze(-1)
        epistemic_uncertainty_per_modality = torch.stack([self.quantification(
            e)[0] for e in evidences_per_modality])
        aleatoric_uncertainty_per_modality = torch.stack([self.quantification(
            e)[1] for e in evidences_per_modality])
        # dc = get_degree_of_conflict(evidences_per_modality, uncertainty_for_dc)
        return loss, output, target, epistemic_uncertainty, aleatoric_uncertainty, aleatoric_uncertainty, evidences_per_modality, epistemic_uncertainty_per_modality

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.criterion.annealing_step += 1

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_loss', np.mean(
            [x[0].detach().cpu().numpy() for x in outputs]), prog_bar=True)
        self.log('val_entropy', torch.cat(
            [x[3] for x in outputs]).mean(), prog_bar=True)
        self.log('val_sigma', torch.cat([x[4]
                 for x in outputs]).mean(), prog_bar=True)
        self.dc = torch.cat([x[5] for x in outputs]).detach().cpu().numpy()
        # self.evidences_per_modality = torch.cat(
        #     [x[6] for x in outputs]).detach().cpu().numpy()
        # self.uncertainty_per_modality = torch.cat(
        #     [x[7] for x in outputs]).detach().cpu().numpy()

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_entropy_epi', torch.cat([x[3] for x in outputs]).mean())
        self.log('test_ale', torch.cat([x[4] for x in outputs]).mean())
        self.aleatoric_uncertainties = torch.cat(
            [x[4] for x in outputs]).detach().cpu().numpy()
        self.epistemic_uncertainties = torch.cat(
            [x[3] for x in outputs]).detach().cpu().numpy()
        self.dc = torch.cat([x[5] for x in outputs]).detach().cpu().numpy()
        self.evidences_per_modality = torch.stack(
            [x[6] for x in outputs]).detach().cpu().numpy()
        self.uncertainty_per_modality = torch.stack(
            [x[7] for x in outputs]).detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
