import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


class Metric:
    """
    Use the values class to calculate various metrics
    """
    def __init__(self, metrics):
        # record the output values and target values of the model for each batch
        self.values = {}
        self.outputs = []
        self.targets = []
        self.losses = []
        self.metrics = metrics

    def accuracy(self):
        self.values['acc'] = accuracy_score(self.targets,self.outputs)
        # calculate the accuracy
        return self.values['acc']

    def update(self, outputs, targets, loss=None):
        # append one batch outputs and targets to all outputs and targets
        if torch.is_tensor(outputs):
            self.outputs += outputs.cpu().detach().tolist()
            self.targets += targets.cpu().detach().tolist()
        else:
            self.outputs += outputs.tolist()
            self.targets += targets.tolist()
        if loss is not None:
            self.losses.append(loss)

    def macro_f1_score(self):
        self.values['macro-f1'] = f1_score(self.targets, self.outputs, average='macro')
        # calculate the macro f1-score
        return self.values['macro-f1']

    def micro_f1_score(self):
        self.values['micro-f1'] = f1_score(self.targets, self.outputs, average='micro')
        # calculate the micro f1-score
        return self.values['micro-f1']

    def weighted_f1_score(self):
        self.values['weighted-f1'] = f1_score(self.targets, self.outputs, average='weighted')
        return self.values['weighted-f1']

    def ck_coe(self):
        self.values['ck'] = cohen_kappa_score(self.targets, self.outputs)
        # calculate the micro f1-score
        return self.values['ck']
    def value(self):
        # 　if one hot code, then transform to ordinary label
        if type(self.targets[0]) is list:
            try:
                self.targets = [self.targets[i].index(1) for i in range(len(self.targets))]
            except ValueError:
                return "unavailable"
        func = {
            'acc': self.accuracy,
            'macro-f1': self.macro_f1_score,
            'micro-f1': self.micro_f1_score,
            'ck': self.ck_coe,
            'weighted-f1': self.weighted_f1_score,
        }
        out = ""
        for m in self.metrics:
            out += f"{m}: {func[m]():.3f}   "
        if len(self.losses) != 0:
            return out + f"loss: {sum(self.losses)/len(self.losses):.4f}"
        else:
            return out
            











