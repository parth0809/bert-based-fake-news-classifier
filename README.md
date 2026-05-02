# BERT-Based Fake News Classification

This repository contains a Jupyter notebook for fake news classification using BERT-based text features and Hugging Face's `BertForSequenceClassification`.

The project works with news titles and predicts a binary label:

- `0` = non-fake
- `1` = fake

## Repository Contents

- `NLP_Assignment.ipynb` - main notebook with data loading, training, evaluation, and interpretability steps
- `NLP_ASSIGNMENT3_REPORT.pdf` - assignment report

## What the Notebook Does

The notebook includes the following stages:

1. Loads and combines the public train, validation, and test TSV files.
2. Selects the `clean_title`, `2_way_label`, and `id` columns.
3. Builds a balanced subset of 5,000 samples total:
   - 2,500 fake
   - 2,500 non-fake
4. Splits the subset into:
   - 80% training
   - 20% test
5. Fine-tunes `bert-base-uncased` for binary classification.
6. Evaluates the model with accuracy, precision, recall, F1-score, and a confusion matrix.
7. Uses Captum's Layer Integrated Gradients to inspect token-level attributions.

## Results

The notebook reports the following test-set results after 3 training epochs:

- Accuracy: `0.817`
- F1 score: `0.8211`
- Precision: `0.8031`
- Recall: `0.84`

Confusion matrix:

```text
[[397 103]
 [ 80 420]]
```

## Requirements

The notebook uses:

- `torch`
- `transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `captum`

If you are running locally, install the dependencies first:

```bash
pip install torch transformers pandas numpy scikit-learn captum jupyter
```

## How to Run

1. Open `NLP_Assignment.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Update the dataset paths if needed.
3. Run the cells from top to bottom.

## Data Paths

The notebook currently reads dataset files from Kaggle-style paths such as:

```text
/kaggle/input/datasets/parthupadhyay89/assignment-3/
```

If you are running the notebook outside Kaggle, replace those paths with the location of your local TSV files.

## Notes

- The notebook includes a custom transformer-style BERT implementation for shape checking and experimentation, but the main training pipeline uses Hugging Face's pretrained `bert-base-uncased`.
- The trained model is saved to `./results` at the end of training.
- Captum is installed inside the notebook before the attribution step.

## License

No explicit license is provided in this repository.
