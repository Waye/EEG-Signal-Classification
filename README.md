# EEG Signal Classification for Emotion Recognition

This project focuses on classifying Electroencephalogram (EEG) signals into different emotional states using machine learning techniques. By leveraging deep learning models, we aim to distinguish various emotional states effectively. The approach includes data pre-processing, feature extraction, and model development using advanced deep learning architectures.

## Project Structure

```
.
├── main/                    # Main experimental scripts and analysis notebooks
│   ├── EDA.ipynb            # Jupyter Notebook for Exploratory Data Analysis
│   ├── Models_Comp.ipynb    # Jupyter Notebook comparing different model results
│   └── ResCNN-TransGRU_exp1.ipynb # Jupyter Notebook for experiment 1 with ResCNN-TransGRU model
├── models/                  # Directory for storing model-related resources
│   ├── model-structure      # Directory containing model structure diagrams and information
│   └── BigData-presentation.pptx # Presentation file detailing the model and approach
├── samples/                 # Directory for storing models performance visualizations
│   ├── Attention-BiLSTM-CNN-loss.png
│   ├── Attention-BiLSTM-CNN-valid-accuracy.png
│   ├── multi-cnn-lstm-loss.png
│   ├── multi-cnn-lstm-valid-accuracy.png
│   ├── ResCNN-TransGRU-loss.png
│   └── ResCNN-TransGRU-valid-accuracy.png
├── environment.yml          # File to set up the required environment with dependencies
├── README.md                # Project documentation file
├── LICENSE                  # License information
├── .gitattributes           # Git configuration attributes
└── .gitignore               # Git ignore file
```

## Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Architecture](#model-architecture)
- [Training and Experimentation](#training-and-experimentation)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

## Notice

In our paper, the term `FSTS` corresponds to `Multi-CNN-LSTM` in the code, while `A-FSTS` corresponds to `Attention-BiLSTM-CNN`. The model `ResCNN-TransGRU` is not mentioned in the paper but is included in the `BigData-presentation.pptx` alongside `FSTS(Multi-CNN-LSTM)` and `A-FSTS(Attention-BiLSTM-CNN)`. 

The [initial model](https://github.com/BluesRockets/EEG-emotion-recognition) represents the original implementation of the `FSTS(Multi-CNN-LSTM)` model.

## Overview

This project aims to classify EEG signals into different emotional states using several deep learning models. It provides tools for data analysis, training, evaluating, and comparing models to improve classification performance.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the required dependencies. You can install the necessary packages using `environment.yml`:

```bash
conda env create -f environment.yml
```

### Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Waye/EEG-Signal-Classification
cd eeg-signal-classification
```

## Exploratory Data Analysis

The initial data analysis and visualization are detailed in `main/EDA.ipynb`. This notebook includes:
- Loading and inspecting the dataset.
- Visualizing EEG signals to identify patterns and differences.

To start data analysis, open the notebook and execute the cells step-by-step:

```bash
jupyter notebook main/EDA.ipynb
```




## Data Preprocessing

Data preprocessing includes several critical steps to ensure the EEG data is ready for training:

#### Outlier Removal Before Filtering
Outliers are removed using Z-score thresholding to prevent filter artifacts.

#### Zero-Phase Filtering
Zero-phase filtering is applied to avoid phase distortions using `filtfilt`.

#### Differential Entropy Computation
A small constant is added to variance to prevent errors during logarithmic computation.

#### Data Reshaping to 10-20 System
EEG channels are mapped to an 8x9 grid to align with the 10-20 system.

Data preprocessing steps are documented in `main/Models_Comp.ipynb` and `ResCNN-TransGRU_exp1.ipynb`. These steps are slightly different for training and test sets preparation, but share a common setup for most stages mentioned above.





## Model Architecture

### Attention-BiLSTM-CNN and Multi-CNN-LSTM
The architectures for various deep learning models, such as Attention-BiLSTM-CNN and Multi-CNN-LSTM, are evaluated and visualized in `main/Models_Comp.ipynb`. These models explore combining convolutional Layers , LSTM and attention Layer for effective feature extraction and sequence modeling.

### ResCNN-TransGRU
The ResCNN-TransGRU model integrates two residual connections within its base convolutional layers. These are followed by two transformer encoder blocks, consisting of multi-head attention and a Feed-Forward Network (FFN). Finally,  the model incorporates a GRU(Gated recurrent unit), to handle both spatial and temporal dynamics effectively. The experiments are documented in `main/ResCNN-TransGRU_exp1.ipynb`.

The **model-structure** directory contains visual representations and additional documentation of the network architectures.

If you want to acquire more details of each fold's model, you can access the following [Model Files](https://drive.google.com/drive/folders/1GbFTjBxWj_xWIuEwDFDA0A57nRt31f4O?usp=drive_link) Download the `.h5` files and open them using the software Netron to visualize the model structure.

## Training and Experimentation

The training experiments are carried out within `main/Models_Comp.ipynb` and `main/ResCNN-TransGRU_exp1.ipynb`. These Jupyter Notebooks demonstrate:
- Model training loops.
- Hyperparameter adjustments.
- Validation and testing results.

You can run these notebooks to observe the training process:

```bash
jupyter notebook main/Models_Comp.ipynb
```

## Evaluation

The training and test results, including accuracy and loss graphs, are stored in the `samples/` directory:
- `Attention-BiLSTM-CNN-valid-accuracy.png` and `loss.png` for Attention-BiLSTM-CNN.
- `multi-cnn-lstm-valid-accuracy.png` and `loss.png` for Multi-CNN-LSTM.
- `ResCNN-TransGRU-valid-accuracy.png` and `loss.png` for ResCNN-TransGRU.


More detailed test results for each fold's performance can be seen from the Google Drive links [TensorBoard logs](https://drive.google.com/drive/folders/1wDcNknsnC4YcHXuGf7TMXk6b82unuHMo)

To visualize these graphs with TensorBoard, run the following command in the terminal:

```bash
tensorboard --logdir=path/to/logs/fit
```

This will start a local server that you can access via a web browser to view the training metrics and graphs interactively.

In addition, `Models_Comp.ipynb` and `ResCNN-TransGRU_exp1.ipynb` contain outputs such as `mean_valid` and `std_vali` for the final validation set.


## Future Improvements

- **Exploration of Additional Features**: Investigating other EEG features or expanding the frequency bands could potentially uncover new patterns that improve emotion recognition accuracy.

- **Real-World Validation**: Applying the proposed model to EEG data collected in real-world, less-controlled environments could demonstrate its robustness and practical applicability.

- **Integration with Other Modalities**: ECombining EEG signals with other physiological or behavioral data (e.g., facial expressions, eye movement) could enhance the overall emotion recognition system.

## Others
- [Report and literature review thesis](https://drive.google.com/drive/folders/1YgNaDM-EeLweEmtAuCXV2IMaL5h8grT1?usp=drive_link) as well as all other implementation files

- [The initial Model](https://github.com/BluesRockets/EEG-emotion-recognition)


## Contributors

- [Weiyi Hu](https://github.com/Waye) 
- [Ruobin Tian](https://github.com/LONGTRBLIVE) 
- [MingHai Liang](https://github.com/BluesRockets) 

## Supervisor
- [Dr. T. Akilan, P.Eng](https://www.lakeheadu.ca/users/A/takilan)