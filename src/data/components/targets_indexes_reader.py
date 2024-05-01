import pickle


def pickle_load(path):
    with open(path, "rb") as input:
        return pickle.load(input)


class IndexesReader:
    def __init__(self, train_path: str = None, val_path: str = None, test_path: str = None) -> None:
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

    @property
    def train_indexes(self):
        if self.train_path is not None:
            return pickle_load(self.train_path)
        else: 
            raise NotImplementedError("train path is None")
            
    
    @property
    def val_indexes(self):
        if self.val_path is not None:
            return pickle_load(self.val_path)
        else: 
            raise NotImplementedError("val path is None")
    
    @property
    def test_indexes(self):
        if self.test_path is not None:
            return pickle_load(self.test_path)
        else: 
            raise NotImplementedError("test path is None")
        


class TargetsReader:
    def __init__(self, targets_path: str) -> None:
        self.targets_path = targets_path

    @property
    def targets(self):
        if self.targets_path is not None:
            return pickle_load(self.targets_path)
        else: 
            raise NotImplementedError("targets path is None")