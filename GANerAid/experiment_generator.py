from GANerAid.ganeraid import GANerAid
from GANerAid.utils import set_or_default


class ExperimentGenerator:
    def __init__(self, device, dataset, experiment_parameters):
        self.device = device
        self.experiment_parameters = experiment_parameters
        self.dataset = dataset

    def execute_experiment(self, verbose=True):
        evaluation_results = []
        for experiment in self.experiment_parameters:
            print("Run experiment {}".format(experiment))
            gan = GANerAid(self.device, **experiment)
            gan.fit(self.dataset, epochs=set_or_default('epochs', 1000, experiment), verbose=verbose)
            data_gen = gan.generate(sample_size=experiment['sample_size'])
            result = gan.evaluate(self.dataset, data_gen)
            single_result = {'parameters': experiment, 'evaluationResult': result}
            evaluation_results.append(single_result)
        return evaluation_results



