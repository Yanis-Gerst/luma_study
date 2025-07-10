import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from octopy.octopy.metrics.conflict.conflict_change_rate import get_degree_of_conflict
from baselines.utils import AvgTrustedLoss
from octopy.octopy.uncertainty.quantification import Dirichlet
from octopy.octopy.uncertainty.loss import EvidentialLoss
from torch.nn import CrossEntropyLoss
from octopy.octopy.uncertainty.layers import EvidentialActivation
from octopy.octopy.uncertainty.loss.unimodal_loss import DECLoss


class UnimodalModel(pl.LightningModule):
    def __init__(self, model, num_classes=42, dropout=0., lr=1e-3, annealing_step=50, activation="exp", clamp_max=10):
        super(UnimodalModel, self).__init__()
        self.num_classes = num_classes
        self.model = model(num_classes=num_classes,
                           monte_carlo=False, dropout=dropout)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.lr = lr
        self.annealing_step = annealing_step
        if activation == "exp":
            self.evidential_activation = EvidentialActivation(
                "exp", clamp_max=clamp_max)
        else:
            self.evidential_activation = EvidentialActivation("softplus")

        self.criterion = EvidentialLoss(
            num_classes=num_classes, device=self.device)
        self.criterion = DECLoss(
            annealing_step=self.annealing_step, loss_type="digamma")
        # self.criterion = CrossEntropyLoss()

        self.quantification = Dirichlet(num_classes)

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
        fused_output = self((image, audio, text))
        fused_output = self.evidential_activation(fused_output)
        loss = self.criterion(fused_output, target, self.current_epoch)
        print("Loss: ", loss)
        # print(fused_output.argmax(dim=1))

        return loss, fused_output, target

    def validation_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(
            batch, True)
        self.val_acc(output, target)

        return loss, output, target

    def test_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(
            batch, True)
        self.test_acc(output, target)

        return loss, output, target

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        # self.criterion.annealing_step += 1

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_loss', np.mean(
            [x[0].detach().cpu().numpy() for x in outputs]), prog_bar=True)
        # self.log('val_entropy', torch.cat(
        #     [x[3] for x in outputs]).mean(), prog_bar=True)
        # self.log('val_sigma', torch.cat([x[4]
        #          for x in outputs]).mean(), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        # self.log('test_entropy_epi', torch.cat([x[3] for x in outputs]).mean())
        # self.log('test_ale', torch.cat([x[4] for x in outputs]).mean())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
