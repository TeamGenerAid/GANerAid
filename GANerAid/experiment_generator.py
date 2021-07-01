from pathlib import Path

from GANerAid.ganeraid import GANerAid
from GANerAid.utils import set_or_default


class ExperimentGenerator:
    def __init__(self, device, dataset, experiment_parameters):
        self.device = device
        self.experiment_parameters = experiment_parameters
        self.dataset = dataset

    def execute_experiment(self, verbose=True, save_models=False, save_path="experiment"):
        evaluation_results = []
        if save_models:
            Path(save_path).mkdir(parents=True, exist_ok=True)
        for experiment in self.experiment_parameters:
            print("Run experiment {}".format(experiment))
            gan = GANerAid(self.device, **experiment)
            gan.fit(self.dataset, epochs=set_or_default('epochs', 1000, experiment), verbose=verbose)
            data_gen = gan.generate(sample_size=experiment['sample_size'])
            result = gan.evaluate(self.dataset, data_gen)
            single_result = {'parameters': experiment, 'evaluationResult': result}
            evaluation_results.append(single_result)
            if save_models:
                model_name = "-".join([key + "_" + str(experiment[key]) for key in experiment.keys()])
                gan.save(path=save_path + "/" + model_name)
        return evaluation_results
