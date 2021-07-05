import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tab_gan_metrics import TableEvaluator
import warnings
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


class EvaluationReport:
    def __init__(self, original_data, generated_data):
        self.original_data = original_data
        self.generated_data = generated_data

    def plot_evaluation_metrics(self):
        print('\n')
        print("EVALUATION REPORT")
        print("----------------------------")
        table_evaluator = TableEvaluator(self.original_data, self.generated_data)
        table_evaluator.visual_evaluation()

    def plot_correlation(self, size_x=60, size_y=120):
        self.plot_original_correlation()
        self.plot_generated_correlation()

    def plot_generated_correlation(self, size_x=60, size_y=120):
        plt.Figure(figsize=(size_x, size_y))
        sns.heatmap(self.generated_data.corr(), cmap="YlGnBu")
        plt.title("Generated Data Correlation Matrix")
        plt.show()

    def plot_original_correlation(self, size_x=60, size_y=120):
        plt.Figure(figsize=(size_x, size_y))
        sns.heatmap(self.original_data.corr(), cmap="YlGnBu")
        plt.title("Real Data Correlation Matrix")
        plt.show()

    def get_correlation_metrics(self):
        print('\n')
        print("CORRELATION METRICS")
        print("----------------------------")
        euclidean_dist = np.linalg.norm(self.original_data.corr().abs() - self.generated_data.corr().abs())
        print("Euclidean Distance {}".format(str(euclidean_dist)))

        if self.original_data.shape != self.generated_data.shape:
            raise ValueError("The RMSE can only be calculated when the datasets ave the same size.")

        for column in self.original_data.columns:
            original_values = self.original_data[column]
            generated_values = self.generated_data[column]
            rmse = mean_squared_error(original_values, generated_values, squared=False)
            print("Root Mean Square Error (RMSE) for Column {}: {}".format(str(column), (str(rmse))))

    def get_duplicates(self):
        print('\n')
        print("DUPLICATES")
        print("----------------------------")
        print("Real dataset contains {} duplicated rows".format(str(self.original_data.duplicated().sum())))
        print("Generated dataset contains {} duplicated rows".format(str(self.generated_data.duplicated().sum())))
        print("Real and generated dataset contain {} duplicated rows".format(str(pd.concat([self.original_data, self.generated_data]).duplicated().sum())))

    def get_KL_divergence(self):
        print('\n')
        print("KULLBACK-LEIBLER DIVERGENCE")
        print("----------------------------")

        if self.original_data.shape != self.generated_data.shape:
            raise ValueError("The clalculation of the Kullback Leibler divergence can only be done if the datasets "
                             "have th same size.")

        for column in self.original_data.columns:
            original_values = self.original_data[column].to_numpy()
            generated_values = self.generated_data[column].to_numpy()
            kl_div = np.sum(
                np.where(original_values != 0, original_values * np.log(original_values / generated_values), 0))
            print("{} : {}".format(column, kl_div))
