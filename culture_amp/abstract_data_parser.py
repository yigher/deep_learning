"""abstract_data_parser script containing abstract base class"""
import abc
class AbstractDataParser(object):
    """AbstractDataParser for training/testing data"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load(self, input_source):
        """load the data from the input_source"""
        return

    @abc.abstractmethod
    def convert(self):
        """Convert the input_source"""
        return

    @abc.abstractmethod
    def save(self, output, data):
        """Save the data object to the output."""
        return
