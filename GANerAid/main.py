from ganeraid import GANerAid
from experiment_generator import ExperimentGenerator
import pandas as pd
import torch

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv('Gen_data_2000.csv').drop(
        ["Caucasian", "Asian", "Unnamed: 0", "id", "From_Asia", "From_North_America", "From_Rest_of_world",
         "From_Western_Europe"], axis=1)

    gan = GANerAid(device)
    gan.fit(data, epochs=5)

    data_gen = gan.generate(sample_size=3)
    print(data_gen)
    print(data_gen.shape)

    evaluation_report = gan.evaluate(data, data_gen)
    evaluation_report.get_duplicates()
    evaluation_report.get_KL_divergence()

    parameters = [{'lr_d': 5e-4, 'epochs': 500, 'sample_size': 5},
                  {'lr_d': 5e-9, 'epochs': 500, 'sample_size': 5}]

    generator = ExperimentGenerator(device, data, parameters)





