from abc import ABC, abstractmethod

class DeblurringMethod(ABC):
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def deblur(self, blurred_image, kernel, **kwargs):
        pass
    
    def get_name(self):
        return self.name
    
    def get_default_params(self):
        return {}