from .wiener import WienerDeconvolution
from .richardson_lucy import RichardsonLucyDeconvolution
from .total_variation import TotalVariationDeblurring

# Dictionary mapping method names to their classes
METHOD_REGISTRY = {
    "wiener": WienerDeconvolution,
    "richardson_lucy": RichardsonLucyDeconvolution,
    "total_variation": TotalVariationDeblurring
}

def get_method(method_name):
    """
    Get a deblurring method instance by name.
    
    Parameters:
    method_name (str): Name of the method (wiener, richardson_lucy, total_variation)
    
    Returns:
    DeblurringMethod: Instance of the requested deblurring method
    """
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown deblurring method: {method_name}. "
                         f"Available methods: {', '.join(METHOD_REGISTRY.keys())}")
    
    return METHOD_REGISTRY[method_name]()

def get_available_methods():
    """
    Get a list of available deblurring methods.
    
    Returns:
    list: List of available method names
    """
    return list(METHOD_REGISTRY.keys())