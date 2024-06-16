from .HabLSTM import *


def get_convrnn_model(name, **kwargs):
    models = {
        'hablstm': get_HabLSTM,
    }
    return models[name.lower()](**kwargs)
