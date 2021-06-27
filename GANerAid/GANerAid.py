# main entry point
import pandas as pd
import numpy as np
from UtilityFunctions import set_or_default, noise
from DataProcessing import DataProcessor
from GanTrainer import GanTrainer

from model import GANerAidGAN


class GANerAid():
    def __init__(self, device, **kwargs):
        self.device = device
        # hyper parameters
        self.lr_d = set_or_default("lr_d", 5e-4, kwargs)
        self.lr_g = set_or_default("lr_g", 5e-4, kwargs)
        self.noise_factor = set_or_default("noise_factor", 5, kwargs)
        self.hidden_feature_space = set_or_default("hidden_feature_space", 200, kwargs)
        self.epochs = set_or_default("epochs", 1000, kwargs)
        self.batch_size = set_or_default("batch_size", 100, kwargs)
        self.nr_of_rows = set_or_default("nr_of_rows", 25, kwargs)
        self.binary_noise = set_or_default("binary_noise", 0.2, kwargs)

        # data processing
        self.processor = None

        # gan
        self.gan = None
        self.fitted = False
        self.noise_size = None

        # dataset parameters
        self.dataset_rows = None
        self.dataset_columns = None
        self.binary_columns = None

        self.dataset = None

    def fit(self, dataset, epochs=1000):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('Dataset is not of type Pandas Dataframe')

        self.processor = DataProcessor(dataset)
        self.dataset = self.processor.preprocess(self.binary_noise)

        gan_trainer = GanTrainer(self.lr_d, self.lr_g)

        self.dataset_columns = dataset.shape[1]
        self.dataset_rows = dataset.shape[0]
        self.noise_size = self.dataset_columns * self.noise_factor
        self.gan = GANerAidGAN(self.noise_size, self.nr_of_rows, self.dataset_columns, self.hidden_feature_space,
                               self.device)

        gan_trainer.train(self.dataset, self.gan, epochs)

        self.fitted = True

        print("fit gan")

    def continue_training(self, epochs=1000):
        # todo: continue training
        print("continue to train gan")

    def generate(self, sample_size=1000):
        if not self.fitted:
            raise ValueError('Gan needs to be fitted by calling fit(dataset) before calling generate()')
        # todo: generate data
        self.gan.eval()
        generate = lambda: self.gan.generator(noise(1, self.noise_size)).view(self.nr_of_rows,
                                                                              self.dataset_columns).cpu().detach()
        sample = generate().numpy()
        for i in range(int(sample_size / self.nr_of_rows) -1):
            sample = np.append(sample,
                               generate().numpy(),
                               axis=0)

        return self.processor.postprocess(sample)

    def evaluate(self):
        if not self.fitted:
            raise ValueError('Gan needs to be fitted by calling fit(dataset) before calling evaluate()')
        # todo: generate evaluation report
        print("evaluate data")
