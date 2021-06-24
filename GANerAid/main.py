from GANerAid import GANerAid
import pandas as pd
if __name__ == '__main__':
    data = pd.read_csv('Gen_data_2000.csv').drop(
        ["Caucasian", "Asian", "Unnamed: 0", "id", "From_Asia", "From_North_America", "From_Rest_of_world",
         "From_Western_Europe"], axis=1)

    gan = GANerAid(lr_d = 1)
    gan.fit(data)
    print(gan.lr_d)
    data = gan.generate(sample_size = 100)
    print(data)
