# GANerAid

GANerAid is a library meant to create synthetic clinical trials in order to anonymise and augment patient data in the field of medical science. The generated data are of the same statistical properties like as the original input data to reproduce the patient data in the best possible way without copying it.

Therefore a Generative Adversarial Network [(GAN)](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/) is used in a way to process and synthesise tabular data, containing continous and binary variables instead of images.

The libray consist of (3) four diffenrent areas:
- Data preprocessing 
- Data generation
- (Model Improvement) 
- Evaluation 


## Installation
The following libraries are needed and asould be installed in advance: 
- numpy>=1.19.5
- pandas>=1.2.4
- torch>=1.8.1
- scikit-learn>=0.22.2
- seaborn>=0.11.1
- tqdm>=4.61.1
- tab-gan-metrics >= 1.1.4

Or simply running the `setup.py` file:

```
python3 setup.py install
```

To install the GANerAid library Python Package Indexing can be used:
```
pip install GANerAid
```


## Provided fuctions and Usage
### Data Preprocessing 

The library offers the possibility to use data augmentation. This feature doubles the data specified as input in order to have double the amount available for the GAN training process.
To use it, simply set the use_aug parameter of the fit() method to true: 

```
gan.fit(data, epochs=5, use_aug=True)
```

By doing so, the GAN will be fitted to the original input data + augmented data with the same size as the original input data.

### Data Generation

After fitting, the generate() method has to be called in order to generate synthetic data. 

```
data = gan.generate(sample_size=100)
```

### Evaluation

### Parameters

| Parameter        | Description           | Default Value  |
| ------------- |-------------| -----|
| lr_d     | Learning rate of the generator | 5e-4|
| lr_g     | Learning rate of the discriminator     |   5e-4 |
| noise_factor | Noise factor defining how large the noise vector as input for the generator will be based on the nr of columns. E.g. 10 columns and a noise factor of 5 results in an input vector with the length of 50       |    5 |


## Dataset
You can simply try it out using Kaggles [Breast cancer dataset](https://www.kaggle.com/merishnasuwal/breast-cancer-prediction-dataset) as input.


## License
All code is under MIT license, except the following. 
