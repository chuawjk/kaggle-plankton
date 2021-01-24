import argparse
import logging
from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf

from src.config import WEIGHTS_DIR, DEFAULT_CLASSES
from src.datapipeline.preprocessing import preprocess
from src.modelling.model import initialise_model

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("inference")

LOG.info("Initialising model")
WEIGHTS_PATH = WEIGHTS_DIR / "default" / "weights.h5"
model = initialise_model()
model.load_weights(WEIGHTS_PATH)


def make_inference(image_path):
    """
    Takes input image and returns prediction probabilities

    Args:
        image_path (str): Path to inference image
    
    Returns:
        Prediction probabilities in descending order of likelihood
    """
    LOG.info("Starting inference")
    
    image = Image.open(Path(image_path))
    image = np.array(image)
    image = preprocess(image)

    pred_probs = model.predict(np.expand_dims(image, 0))
    highest_prob = np.max(pred_probs)
    index = np.where(pred_probs == highest_prob)[1][0]
    pred_class = DEFAULT_CLASSES[index]

    LOG.info("Inference completed")
    return {pred_class: highest_prob}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image")
    parser.add_argument("--weights_path")
    args = parser.parse_args()

    if args.weights_path:
        LOG.info("Initialising model")
        model = initialise_model()
        model.load_weights(Path(args.weights_path))

    pred_dict = make_inference(Path(args.input_image))
    print(pred_dict)
