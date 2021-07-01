# GANerAid

![GenerAid](generaid.png)

GANerAid is a library meant to create synthetic clinical trials in order to anonymise and augment patient data in the field of medical science. The generated data are of the same statistical properties like as the original input data to reproduce the patient data in the best possible way without copying it.

Therefore a Generative Adversarial Network [(GAN)](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/) is used in a way to process and synthesise tabular data, containing continous and binary variables instead of images.

The libray consist of (3) four diffenrent areas:
- Data preprocessing 
- Data generation
- Parameter optimisation
- Evaluation 


## Installation
The following libraries are needed: 
- numpy>=1.19.5
- pandas>=1.2.4
- torch>=1.8.1
- scikit-learn>=0.22.2
- seaborn>=0.11.1
- tqdm>=4.61.1
- tab-gan-metrics >= 1.1.4
- matplotlib>=3.4.0

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

For the dataset to be passed to the library, the following must be ensured:
- Binary variables must be converted to 0 and 1
- The dataset must be passed in a pandas dataframe
- Categorical variables should be avoided and therefore removed

Internally the library is then further preprocessing the data. The data gets scaled to the range between -1 and 1. Afterwards, as small amount of noise is added to the binary columns. The data augmentation discussed in the next paragrahp is also handled here.

### Parameters

The hyper parameters, such as learning rate, can be set in the constructor call of the GanerAid class.
```
gan = GANerAid(device, lr_d=0.000005, lr_g=0.000005)
```

| Parameter        | Description           | Default Value  |
| ------------- |-------------| -----|
| lr_d     | Learning rate of the generator | 5e-4|
| lr_g     | Learning rate of the discriminator     |   5e-4 |
| noise_factor | Noise factor defining how large the noise vector as input for the generator will be based on the nr of columns. E.g. 10 columns and a noise factor of 5 results in an input vector with the length of 50       |    5 |
| hidden_feature_space | The feature space of the LSTm Cells| 200|
| batch_size | The batch size defines the number of samples that will be propagated through the network. | 100 |
| batch_size | The batch size of how many rows the generator generates at once. A lower number will make the correlation better. A higher number will reproduce the distributions of the colujmsn better. | 25 |
| binary_noise | The upper limit of the uniformly distributed noise that will be added to binary columns in the preprocessing. | 0.2 |

### Data Augmentation

The library offers the possibility to use data augmentation. This feature increases the data size based on a given parameter specified as input in order to have more data available for the GAN training process.
To use it, simply set the aug_factor parameter of the fit() method to a value > 0. <br>
This will increase the data based on the following formula data_size = original_data_size + (original_data_size * aug_factor) <br>
E.g an aug_factor of 1 will double the value. 

```
gan.fit(data, epochs=5, aug_factor=1)
```

By doing so, the GAN will be fitted to the original input data + augmented data with the same size as the original input data.

### Data Postprocessing

After the GAN has generated data it needs to be postprocessed. This is done internally analogous to the preprocessing. The preprocessing steps are reversed, i.e. the values of the binary columns are set to -1 or 1 according to the interpretation of the noisy values. Afterwards the complete dataset gets scaled back to its original range. For Integer columns the values are being rounded afterwards. 


### Data Generation

After fitting, the generate() method has to be called in order to generate synthetic data. The returned data is already postprocessed.

```
data = gan.generate(sample_size=100)
```
### Parameter optimisation

The library offers the possibility to define several parameter combinations and to run them one after the other. At the end, these are returned with the corresponding evaluation results and can be analyzed.

For this, the parameter combinations must be defined in Python dictionaries and passed to the library function in a list at the end:
```
#define parameters
parameters = [{'lr_d': 5e-4, 'epochs': 500, 'sample_size': 5},
              {'lr_d': 5e-9, 'epochs': 500, 'sample_size': 5}]

#create exepriment runner instance
generator = ExperimentGenerator(device, data, parameters)

#execute experiments and save trained models
generator.execute_experiment(save_models=True, save_path="experiment")

```

Valid parameters which can be specified in the dictionaries are the ones listed above in the 'Parameters' section.

### Evaluation

Some evaluation functions are provided in the library. Hence, it can be decided whether the GAN generated satisfactory data.
For this purpose, approximately the same number of data to be evaluated should be generated as the data set originally passed to the library.

To be able to use the fuctions of the valuation just use the method:
```
evaluation_report = gan.evaluate(data, data_gen)
```
where data is the original data and data_gen is the generated data.
#### plot_evaluation_metrics
This method uses the [table_evaluator library](https://github.com/topics/table-evaluator) to provide plots of the ditributions, as well as the correlation of the two datasets.
```
evaluation_report.plot_evaluation_metrics()
```

#### plot_correlation
This method plots the correlation matrices of the original data and the generated data.
```
evaluation_report.plot_correlation()
```
The optional parameters size_x and size_y can be can be used to adjust the size of the plots.

#### get_correlation_metrics
This method prints the euclidean distance of the correlation matrices. The RMSE for each value is prited as well. 
```
evaluation_report.get_correlation_metrics()
```
#### get_duplicates
Thie method prints the number of duplicate rows in the real dataset, the generated dataset and in both datasets.
```
evaluation_report.get_duplicates()
```
#### get_KL_divergence
This method prints the Kullback-Leibler divergence, which is a metric of how similar two ditributions are. The method will compare the two datsets column-wise. Lower values mean more similar distributions. More to Kullback-Leibler divergence can be found [here](https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810)


## Dataset
You can simply try it out using Kaggles [Breast cancer dataset](https://www.kaggle.com/merishnasuwal/breast-cancer-prediction-dataset) as input.


## License
All code is under MIT license. 
