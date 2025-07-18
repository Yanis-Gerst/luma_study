import torch

from baselines.utils import MCDropout
from octopy.octopy.uncertainty.fusion import DiscountedBeliefFusion
from octopy.octopy.uncertainty.layers import EvidentialActivation


class ImageClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.3, monte_carlo=False, aleatoric=False):
        super(ImageClassifier, self).__init__()
        self.image_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(64 * 6 * 6, num_classes)
        self.monte_carlo = monte_carlo
        self.aleatoric = aleatoric
        if monte_carlo or aleatoric:
            self.sigma = torch.nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        image, audio, text = x
        image = self.image_model(image.float())
        if self.monte_carlo or self.aleatoric:
            return self.classifier(image), torch.nn.functional.softplus(self.sigma(image))
        return self.classifier(image)


class AudioClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5, monte_carlo=False, aleatoric=False):
        super(AudioClassifier, self).__init__()
        self.audio_model = torch.nn.Sequential(  # from batch_size x 1 x 128 x 128 spectrogram
            torch.nn.Conv2d(1, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Flatten()
        )
        self.classifier = torch.nn.Linear(64 * 14 * 14, num_classes)
        self.monte_carlo = monte_carlo
        self.aleatoric = aleatoric
        if monte_carlo or aleatoric:
            self.sigma = torch.nn.Linear(64 * 14 * 14, num_classes)

    def forward(self, x):
        image, audio, text = x
        audio = self.audio_model(audio)
        if self.monte_carlo or self.aleatoric:
            return self.classifier(audio), torch.nn.functional.softplus(self.sigma(audio))
        return self.classifier(audio)


class TextClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5, monte_carlo=False, aleatoric=False):
        super(TextClassifier, self).__init__()
        self.text_model = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
        )
        self.classifier = torch.nn.Linear(256, num_classes)
        self.monte_carlo = monte_carlo
        self.aleatoric = aleatoric
        if monte_carlo or aleatoric:
            self.sigma = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        image, audio, text = x
        text = self.text_model(text)
        if self.monte_carlo or self.aleatoric:
            return self.classifier(text), torch.nn.functional.softplus(self.sigma(text))
        return self.classifier(text)


class MultimodalClassifier(torch.nn.Module):
    def __init__(self, num_classes, id, dropout=0.5, monte_carlo=False, dirichlet=False, aleatoric=False, clamp_max=10, activation="exp"):
        super(MultimodalClassifier, self).__init__()
        self.image_model = ImageClassifier(
            num_classes, dropout, monte_carlo, aleatoric)
        self.audio_model = AudioClassifier(
            num_classes, dropout, monte_carlo, aleatoric)
        self.text_model = TextClassifier(
            num_classes, dropout, monte_carlo, aleatoric)

        self.audio_model.load_state_dict(torch.load(
            f"unimodal_weights/audio_{id}.pth"))
        self.text_model.load_state_dict(torch.load(
            f"unimodal_weights/text_{id}.pth"))
        self.image_model.load_state_dict(torch.load(
            f"unimodal_weights/image_{id}.pth"))
        self.num_views = 3
        self.fusion = DiscountedBeliefFusion(self.num_views, num_classes)
        self.activation = activation

        if self.activation == "s-exp" or self.activation == "exp":
            self.evidential_activation = EvidentialActivation(
                "exp", clamp_max=clamp_max)
        else:
            self.evidential_activation = EvidentialActivation(
                activation)
        self.monte_carlo = monte_carlo
        self.dirichlet = dirichlet
        self.aleatoric = aleatoric
        if dirichlet and monte_carlo:
            raise ValueError(
                "Dirichlet and Monte Carlo cannot be used together")

    def forward(self, x):
        image_outputs = self.image_model(x)
        audio_outputs = self.audio_model(x)
        text_outputs = self.text_model(x)

        if self.monte_carlo or self.aleatoric:
            image_logits, image_sigma = image_outputs
            audio_logits, audio_sigma = audio_outputs
            text_logits, text_sigma = text_outputs
            logits = (image_logits + audio_logits + text_logits) / 3
            sigma = (image_sigma + audio_sigma + text_sigma) / 3
            return logits, sigma
        elif self.dirichlet:
            if self.activation == "s-exp":
                image_logits = torch.nn.functional.softplus(image_outputs)
                audio_logits = torch.nn.functional.softplus(audio_outputs)
                text_logits = torch.nn.functional.softplus(text_outputs)
                image_logits = self.evidential_activation(image_logits)
                audio_logits = self.evidential_activation(audio_logits)
                text_logits = self.evidential_activation(text_logits)
            else:
                image_logits = self.evidential_activation(image_outputs)
                audio_logits = self.evidential_activation(audio_outputs)
                text_logits = self.evidential_activation(text_outputs)

            # print(image_logits.argmax(dim=1), audio_logits.argmax(
            #     dim=1), text_logits.argmax(dim=1))
            evidences = dict()
            evidences[0] = image_logits
            evidences[1] = audio_logits
            evidences[2] = text_logits
            logits = self.fusion(evidences)
            # logits = ((image_logits + audio_logits) / 2 + text_logits) / 2
            # print(logits.argmax(dim=1))
            return logits, evidences
        logits = (image_outputs + audio_outputs + text_outputs) / 3
        return logits
