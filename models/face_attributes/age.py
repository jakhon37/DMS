import os
import gdown
import numpy as np
import deepface

import deepface.DeepFace

print("All attributes and methods in 'deepface.DeepFace':", dir(deepface.DeepFace))

has_class = hasattr(deepface.DeepFace, 'DeepFace')
print(f"Does 'deepface.DeepFace' have the class 'DeepFace'? {has_class}")

has_method = hasattr(deepface.DeepFace, 'analyze')
print(f"Does 'deepface.DeepFace' have the method 'analyze'? {has_method}")



# Print the file path of the imported module
print("Module file path:", deepface.__file__)

# List all attributes and methods
all_attributes = dir(deepface)
print("All attributes and methods in 'deepface':", all_attributes)

# Example: Check if 'DeepFace' class and 'analyze' method are in the module
class_name = 'DeepFace'
method_name = 'analyze'

has_class = hasattr(deepface, class_name)
print(f"Does 'deepface' have the class '{class_name}'? {has_class}")

has_method = hasattr(deepface, method_name)
print(f"Does 'deepface' have the method '{method_name}'? {has_method}")


# from deepface.basemodels import VGGFace
# from deepface.commons import package_utils, folder_utils
# from deepface.models.Demography import Demography
# from deepface.commons import logger as log

logger = deepface.DeepFace.deepface.commons.logger.get_singletonish_logger()

# ----------------------------------------
# dependency configurations

tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

# ----------------------------------------

# pylint: disable=too-few-public-methods
class ApparentAgeClient(deepface.models.Demography):
    """
    Age model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Age"

    def predict(self, img: np.ndarray) -> np.float64:
        age_predictions = self.model.predict(img, verbose=0)[0, :]
        return find_apparent_age(age_predictions)


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5",
) -> Model:
    """
    Construct age model, download its weights and load
    Returns:
        model (Model)
    """

    model = VGGFace.base_model()

    # --------------------------

    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    age_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights

    home = folder_utils.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/age_model_weights.h5") != True:
        logger.info("age_model_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/age_model_weights.h5"
        gdown.download(url, output, quiet=False)

    age_model.load_weights(home + "/.deepface/weights/age_model_weights.h5")

    return age_model

    # --------------------------


def find_apparent_age(age_predictions: np.ndarray) -> np.float64:
    """
    Find apparent age prediction from a given probas of ages
    Args:
        age_predictions (?)
    Returns:
        apparent_age (float)
    """
    output_indexes = np.array(list(range(0, 101)))
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age