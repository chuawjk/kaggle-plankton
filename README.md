# Plankton Classification
Based on the Kaggle [plankton classification problem](https://www.kaggle.com/c/datasciencebowl/overview/evaluation).

Disclaimer: This code was written in about one day. Given the time constraint, the model should be considered a baseline reference. Model performance could be improved if given more engineering time (see last section on future work).

---
## Instructions for use
### Installation
1. Use the terminal to clone the git repositry:
```
git clone https://github.com/chuawjk/kaggle-plankton.git
```
2. Download the trained [`weights.h5`](https://drive.google.com/file/d/1-JI7S1m3BJ31h_UWRSC7QXU65OC6F67P/view?usp=sharing) and place it in the `weights/default` folder.
3. Use the terminal to navigate to the directory containing the `src` folder:
```
cd <path to directory>
```
4. To install the necessary libraries, create a virtual environment from `src/conda.yml` by running the following:
```
conda env create -f conda.yml
```
5. Once done, activate the newly created environment:
```
conda activate kaggle_plankton
```

### Inference
Perform inference by running `src/inference.py` from the terminal:
```
python -m src.inference --input_image <path to input image>
```
This returns the predicted plankton type and its probability as a dictionary, for example:
```
{'amphipods': 0.7489151}
```

### Configuration
Variables including file directories, default hypeparameters and data classses can be edted via `src/config.py`.

---
## Data exploration and model development
See `notebooks/dev_notebook.ipynb` for detailed records on data exploration and model development. Development was performed using compute resources on Google Colab.
### Dataset
The dataset comprises about 24,000 images across 121 plankton labels. There is also a larger, unlabelled test set, presumably for Kaggle evaluation purposes.

The images are in single-channel greyscale format, with lengths or widths up to 400 px, and varying aspect ratios. There is a noticeable imbalance in class frequencies. Visual inspection suggests that the images are of comparable brightness and contrast.

Each image contains an uncropped whole-body view of a single plankton individual. Each class can contain images depicting the organisms in different poses/orientations.

An 80:20 train-val split was performed, and the split was stratified by class to ensure comparable class frequencies in both data subsets.

### Preprocessing
The data pipeline comprises three steps. First, the image is padded along the minor axis to match the major axis. This helps to maintain a proportional depiction of the organism. Next, the image is resized to 224*224. Finally, the greyscale image is converted to RGB. The resulting 224\*224\*3 image conforms to the input requirements for the downstream convolutional neural network (CNN).

Before ingestion into the CNN, the image augmentation is performed. This involves random horizontal and vertical flips, and rotations to simulate the organisms in different orientations.

### Model
The CNN is based on a pretrained MobileNet V2 architecture, which was chosen for having been trained on the large, general ImageNet dataset, and for being relatively fast to train and infer from. The convolutional layers are connected to downstream fully-connected layers, which then terminate in 121 output nodes with softmax activations.

Owing to time constraints, the model was trained for 10 epochs with "rule-of-thumb" hyperparameters (see `config.py`). Only weights from the fully-connected layers were made trainable. Training was performed using categorical cross-entropy loss (appropriate for multi-label classification) and the Adam optimiser. Class weights were set proportional to inverse class frequencies to address data imbalance.

### Evaluation
Performance of this baseline model is as yet indequate, with a mean per-class accuracy of 0.008 (SD 0.091).

### Future work
The current MobileNet-based architecture will likely perform better with more training epochs and a full hyperparameter sweep.

[Literature](https://doi.org/10.1016/j.ecoinf.2019.02.007) suggests that DenseNet architectures may yield greater performance for plankton classification tasks.

Lastly, implementation of a dual-stage model incorporating domain knowledge may also produce improvements in classification performance. Specifically, this could involve an initial coarse classification to higher taxonomic/functional groups (e.g., "shrimps"), followed by a second neural network to perform finer-grained classification to the actual labels required for this task (e.g., "caridean_shrimp').