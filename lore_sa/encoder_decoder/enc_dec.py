from abc import abstractmethod
from lore_sa.dataset.dataset import Dataset

import pandas as pd

__all__ = ["EncDec"]
class EncDec():
    """
    Generic class to implement an encoder/decoder

    It is implemented by different classes, each of which must implements the functions: enc, dec, enc_fit_transform
    the idea is that the user sends the complete record and here only the categorical variables are handled
    """
    def __init__(self,):
        self.dataset_encoded = None
        self.original_features = None
        self.original_data = None
        self.encoded_features = None
        self.original_features_encoded = None
        

    @abstractmethod
    def encode(self, x: Dataset, features_to_encode):
        """
        It applies the encoder to the input features

        :param[Dataset] x: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        """
        return

    @abstractmethod
    def decode(self, x: Dataset, kwargs=None):
        return
