from ganeraid import GANerAid


class ExperimentGenerator:
    def __init__(self, device, dataset, experimentParameters):
        self.device = device
        self.experimentParameters = experimentParameters
        self.dataset = dataset

    def executeExperiment(self):
        evaluationResults = []
        for experiment in self.experimentParameters:
            gan = GANerAid(self.device, experiment)
            gan.fit(self.dataset, epochs=experiment['epochs'])
            data_gen = gan.generate(sample_size=experiment['sample_size'])
            evaluationResult = gan.evaluate(self.dataset, data_gen)
            singleResult = {'parameters': experiment, 'evaluationResult': evaluationResult}
            evaluationResults.append(singleResult)
        return evaluationResults



