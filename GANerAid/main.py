
import pandas as pd
import torch

from GANerAid.experiment_generator import ExperimentGenerator
from GANerAid.ganeraid import  GANerAid

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv('data.csv').drop(
        ["id", "Unnamed: 32", "diagnosis"], axis=1)


    parameters = [{'lr_d': 5e-4, 'epochs': 20, 'sample_size': 5},
                  {'lr_d': 5e-9, 'epochs': 20, 'sample_size': 5}]

    generator = ExperimentGenerator(device, data, parameters)
    generator.execute_experiment(save_models=True, save_path="experiment")
