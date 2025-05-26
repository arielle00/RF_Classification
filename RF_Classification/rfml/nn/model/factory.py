"""Simplistic factory pattern for config defined swapping of architecture used.
"""

# Internal Includes
from .cldnn import CLDNN
from .cnn import CNN
from .resnet_CT import resnet18,resnetx
from .conv5 import TinyConv
from .vgg16 import VGG16
from .transformer import transformerBERT

def build_model(model_name: str, input_samples: int, num_classes: int):
    """Factory method for dynamic creation of multiple neural architectures.

    Args:
        model_name (str): The name of the model to build.  Currently
                          supported models are:
                            - "CNN"
                            - "CLDNN"
        input_samples (int): Number of complex input samples to the model.
        n_classes (int): Number of output classes the model should predict.

    Returns:
        Model: The built model described by the provided parameters
    """
    if input_samples <= 0:
        raise ValueError(
            "The model must take in at least one sample for the input, not {}".format(
                input_samples
            )
        )
    if num_classes <= 2:
        raise ValueError(
            "The models built by this method are for multi-class classification and "
            "therefore must have at least 3 classes to predict from, not {}".format(
                num_classes
            )
        )

    if model_name.upper() == "CNN":
        return CNN(input_samples=input_samples, num_classes=num_classes)
    elif model_name.upper() == "CLDNN":
        return CLDNN(input_samples=input_samples, num_classes=num_classes)
    elif model_name.upper() == "RESNET18":
        return resnet18(input_samples=input_samples, num_classes=num_classes)
    elif model_name.upper() == "CONV5":
        return resnetx(input_samples=input_samples, num_classes=num_classes)
    elif model_name.upper() == "VGG16":
        return VGG16(input_samples=input_samples, num_classes=num_classes)
    elif model_name.upper() == "TRANSFORMERBERT":
        return transformerBERT(input_samples=input_samples, num_classes=num_classes)
    else:
        raise ValueError("Unknown neural network architecture ({})".format(model_name))
