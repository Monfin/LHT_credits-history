import pickle
import lz4
import lz4.frame

from typing import Dict

import os

def serialize_and_compress(item: Dict):
    serialized_item = pickle.dumps(item)
    compressed = lz4.frame.compress(serialized_item)
    return compressed

def deserialize_and_decompress(item):
    decompressed = lz4.frame.decompress(item)
    deserialized = pickle.loads(decompressed)
    return deserialized


class DiskFile:
    def __init__(self, data_path: str, indexes_path: str):
        self.data_path = data_path
        self.indexes_path = indexes_path

        with open(self.indexes_path, "rb") as indexes:
            self.indexes = deserialize_and_decompress(indexes.read())

        self.data = open(data_path, "rb")
            

    def __getitem__(self, idx):
        memory_position, sizeof = self.indexes[idx]
        self.data.seek(memory_position)
        
        item = deserialize_and_decompress(self.data.read(sizeof))
        
        return item
    

class DataReader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def setup(self):
        self.disk_file_data = DiskFile(
            os.path.join(self.data_path, "data.txt"), 
            os.path.join(self.data_path, "index.txt")
        )

    def __contains__(self, idx) -> bool:
        return True if idx in self.disk_file_data.indexes.keys() else False

    def __getitem__(self, idx) -> Dict:
        return self.disk_file_data[idx]
    
    def __len__(self):
        return len(self.disk_file_data.indexes)