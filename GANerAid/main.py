from ganeraid import GANerAid
import pandas as pd
import torch

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv('../heart_failure_clinical_records_dataset.csv').dropna()

    gan = GANerAid(device)
    gan.fit(data, epochs=10000, aug_factor=2)
    
    data_gen = gan.generate(sample_size=data.shape[0])
    print(data_gen)
    print(data_gen.shape)

    evaluation_report = gan.evaluate(data, data_gen)
    evaluation_report.get_duplicates()
    evaluation_report.get_KL_divergence()
    evaluation_report.plot_evaluation_metrics()





