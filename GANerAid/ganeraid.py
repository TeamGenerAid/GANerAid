# main entry point
import pandas as pd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt

from GANerAid.utils import set_or_default, noise
from GANerAid.data_preprocessor import DataProcessor
from GANerAid.gan_trainer import GanTrainer
from GANerAid.evaluation_report import EvaluationReport
from pathlib import Path

from GANerAid.model import GANerAidGAN
from GANerAid.logger import Logger

import logging


class GANerAid:
    def __init__(self, device, **kwargs):
        self.device = device
        self.kwargs = kwargs
        # hyper parameters
        self.lr_d = set_or_default("lr_d", 5e-4, kwargs)
        self.lr_g = set_or_default("lr_g", 5e-4, kwargs)
        self.noise_factor = set_or_default("noise_factor", 5, kwargs)
        self.hidden_feature_space = set_or_default("hidden_feature_space", 200, kwargs)
        self.batch_size = set_or_default("batch_size", 100, kwargs)
        self.nr_of_rows = set_or_default("nr_of_rows", 25, kwargs)
        self.binary_noise = set_or_default("binary_noise", 0.2, kwargs)

        self.logger = Logger(active=set_or_default("logging_activated", True, kwargs))

        # data processing
        self.processor = None

        # gan
        self.gan = None
        self.gan_trainer = None
        self.fitted = False
        self.noise_size = None

        # dataset parameters
        self.orig_dataset = None
        self.dataset_rows = None
        self.dataset_columns = None

        self.dataset = None

        self.logger.print("Initialized gan with the following parameters: \n"
                          "lr_d = {}\n"
                          "lr_g = {}\n"
                          "hidden_feature_space = {}\n"
                          "batch_size = {}\n"
                          "nr_of_rows = {}\n"
                          "binary_noise = {}",
                          self.lr_d,
                          self.lr_g,
                          self.hidden_feature_space,
                          self.batch_size,
                          self.nr_of_rows,
                          self.binary_noise)

    def fit(self, dataset, epochs=1000, verbose=True, aug_factor=0):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('Dataset is not of type Pandas Dataframe')

        if not self.fitted:
            self.orig_dataset = dataset
            self.processor = DataProcessor(dataset)
            self.dataset = self.processor.preprocess(self.binary_noise, aug_factor=aug_factor)

            self.gan_trainer = GanTrainer(self.lr_d, self.lr_g)
            self.logger.print("Start training of gan for {} epochs", epochs)
        else:
            self.logger.print("Continue training of gan for {} epochs", epochs)

        self.dataset_columns = dataset.shape[1]
        self.dataset_rows = dataset.shape[0]
        self.noise_size = self.dataset_columns * self.noise_factor
        self.gan = GANerAidGAN(self.noise_size, self.nr_of_rows, self.dataset_columns, self.hidden_feature_space,
                               self.device)

        history = self.gan_trainer.train(self.dataset, self.gan, epochs, verbose=verbose)

        self.fitted = True

        return history

    def generate(self, sample_size=1000):
        self.logger.print("Generating {} samples", sample_size)

        if not self.fitted:
            raise ValueError('Gan needs to be fitted by calling fit(dataset) before calling generate()')
        self.gan.eval()
        generate = lambda: self.gan.generator(noise(1, self.noise_size)).view(self.nr_of_rows,
                                                                              self.dataset_columns).cpu().detach()
        sample = generate().numpy()
        for i in range(math.ceil(sample_size / self.nr_of_rows) - 1):
            sample = np.append(sample,
                               generate().numpy(),
                               axis=0)

        return self.processor.postprocess(sample[:sample_size])

    def evaluate(self, initial_data, generated_data):
        if not self.fitted:
            raise ValueError('Gan needs to be fitted by calling fit(dataset) before calling evaluate()')
        return EvaluationReport(initial_data, generated_data)

    def save(self, path, name="GANerAid"):
        if not self.fitted:
            raise ValueError('Gan needs to be fitted by calling fit(dataset) before being able to save the gan')
        Path(path).mkdir(parents=True, exist_ok=True)
        gan_params = self.gan.get_params()
        torch.save({
            "gan_params": gan_params,
            "dataset": self.orig_dataset,
            "kwargs": self.kwargs
        }, path + "/" + name + ".gan")
        self.logger.print("Gan successfully saved under the path {} and the name {}", path, name)

    def plot_history(self, history):
        plt.plot(list(zip(history["d_loss"], history["g_loss"])))
        plt.ylabel('Value')
        plt.title('Discriminator and Generator Loss')
        plt.grid(True)
        plt.show()

    @staticmethod
    def load(device, path, name="GANerAid"):
        restored = torch.load(path + "/" + name + ".gan")
        gan = GANerAid(device, **restored["kwargs"])
        gan.fit(restored["dataset"], epochs=0, verbose=False)
        gan.gan = GANerAidGAN.setup_from_params(restored["gan_params"], device)
        return gan
