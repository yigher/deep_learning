"""abstract_model script containing abstract base class"""
import abc
class AbstractModel(object):
    """AbstractModel for training/testing data"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, inputs):
        """fit method"""
        return

    @abc.abstractmethod
    def predict(self, inputs):
        """predict"""
        return

    @abc.abstractmethod
    def save(self, file_location):
        """save model method"""
        return

    @abc.abstractmethod
    def load(self, file_location):
        """load model method"""
        return
