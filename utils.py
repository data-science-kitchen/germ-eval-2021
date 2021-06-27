import numpy as np
from sklearn.preprocessing import LabelEncoder


def multilabel_to_multiclass(labels: np.array) -> np.array:
    return LabelEncoder().fit_transform([''.join(str(label)) for label in labels])
