from pathlib import Path

from GANerAid.ganeraid import GANerAid
from GANerAid.utils import set_or_default
from GANerAid.logger import Logger


class ExperimentRunner:
    def __init__(self, device, dataset, experiment_parameters, logging_activated=True):
        self.device = device
        self.experiment_parameters = experiment_parameters
        self.dataset = dataset
        self.logger = Logger(logging_activated)

    def execute_experiment(self, verbose=True, save_models=False, save_path="experiment"):
        evaluation_results = []
        if save_models:
            Path(save_path).mkdir(parents=True, exist_ok=True)
        for experiment in self.experiment_parameters:
            self.logger.print("Running experiment {}", experiment)
            gan = GANerAid(self.device, **experiment, logging_activated=self.logger.active)
            gan.fit(self.dataset, epochs=set_or_default('epochs', 1000, experiment), verbose=verbose)
            data_gen = gan.generate(sample_size=experiment['sample_size'])
            result = gan.evaluate(self.dataset, data_gen)
            single_result = {'parameters': experiment, 'evaluationResult': result}
            evaluation_results.append(single_result)
            if save_models:
                model_name = "-".join([key + "_" + str(experiment[key]) for key in experiment.keys()])
                gan.save(path=save_path + "/" + model_name)
        return evaluation_results
