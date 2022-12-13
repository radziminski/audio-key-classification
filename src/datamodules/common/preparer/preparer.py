import abc

# Abstract class used for creating preparators that prepare (create/download) a given dataset
class Preparer(abc.ABC):
    # Function that will be called in datamodule prepare
    @abc.abstractmethod
    def prepare(self):
        pass
