# lm-dataset-distillation

## Repository Structure

```
📦 Repository
 ┣ 📂 networks
 ┃ ┗ 📄 (contains boilerplate code for extracting and detaching model weights)
 ┣ 📂 utils
 ┃ ┗ 📄 (includes utilities for filesystem and I/O operations)
 ┣ 📄 distillation_trainer.py
 ┗ 📄 train_notebook.ipynb
```

**distillation_trainer.py**: Performs gradient updates with respect to the data.

**train_notebook.ipynb**: Main notebook for training.

## Running

To perform dataset distillation, open `train_notebook.ipynb` and run all cells in the notebook.

This uses the GPT2 model to create a synthetic dataset, then evaluates the model on both the distilled dataset and the random subsampled dataset. The same evaluation is performed on a separate LSTM model.
