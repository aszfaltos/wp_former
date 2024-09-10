from .model_loader import load_model

class Evaluator:
    def __init__(self):
        self.models = {}
        self.evaluations = {}
        self.metrics = {}

    def add_model(self, path, epoch, load_fn, key):
        params, model, metrics = load_model(path, epoch, load_fn)
        self.metrics[key] = metrics
        self.models[key] = model
        return params

    def rename_model(self, key, new_key):
        self.models[new_key] = self.models[key]
        del self.models[key]
        self.metrics[new_key] = self.metrics[key]
        del self.metrics[key]
        if key in self.evaluations.keys():
            self.evaluations[new_key] = self.evaluations[key]
            del self.evaluations[key]

    def remove_model(self, key):
        del self.models[key]
        del self.metrics[key]
        self.evaluations.pop(key, None)

    def list_models(self):
        return self.models.keys()

    def eval_model_on_dataset(self, key, dataset):
        pass

    def eval_models_on_dataset(self, dataset):
        pass

    def generate_rolling_forecast(self, key, dataset):
        pass

    def generate_rolling_forecasts(self, dataset):
        pass

    def generate_evaluation_table(self):
        pass
