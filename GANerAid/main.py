
import pandas as pd
import torch

from GANerAid.ganeraid import  GANerAid

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv('data.csv').drop(
        ["id", "Unnamed: 32", "diagnosis"], axis=1)


    gan = GANerAid(device)

    gan.fit(data, epochs=100)

    gen_data = gan.generate(500)

    report = gan.evaluate(data, gen_data)

    report.plot_evaluation_metrics()


    print(data)

