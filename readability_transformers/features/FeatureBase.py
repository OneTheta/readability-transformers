from abc import ABC, abstractmethod

class FeatureBase(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def extract(self, text: str) -> dict:
        pass

class CustomFeatureExtractor:
    def __init__(self, name: str, extraction_function):
        self.name = name
        self.extraction_function
    def extract(self, text, options=None):
        if options is not None:
            return self.extraction_function(text, options)
        
        return self.extraction_function(text)
    