# 🥇 Data Science Kitchen at GermEval 2021

This repository contains the codebase and resources for the paper [Data Science Kitchen at GermEval 2021: A Fine Selection of Hand-Picked Features, Delivered Fresh from the Oven](https://arxiv.org/abs/2109.02383), which was submitted to the GermEval 2021 shared task. The task focuses on identifying comments that are toxic, engaging, and fact-claiming, with the goal of helping moderators and community managers prioritize content for fact-checking and engagement.

## 📁 Repository Structure

- **`config`**: Includes training configuration files in YAML format, which can be used to train the model on different feature sets.
- **`data`**: Contains the GermEval 2021 dataset for training and evaluation, as well as pretrained weights of the [AdHominem](https://arxiv.org/abs/1910.08144) model.
- **`src`**: The codebase, including functions for data preparation, model training and evaluation.

## 👷‍♀️ Installation

To run the code in this repository, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/data-science-kitchen/germ-eval-2021.git
   cd germ-eval-2021
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

We have provided a single script `run_pipeline.py`, which performs the entire process of data preparation, feature extraction, model training, tuning and evaluation via a single function call. To reproduce the results from the paper, simply run the following command:

```bash
python run_pipeline.py data/GermEval21_Toxic_Train.csv data/GermEval21_Toxic_TestData.csv config/all-features.yaml
```

Please refer to the corresponding function documentation for additional arguments you can provide to this function.

## 📑 Citation

If you use the code in this repository, please cite our paper:

```
@inproceedings{germeval2021-dsk,
    abbr = {KONVENS},
    author = {N. Hildebrandt and B. Bönninghoff and D. Orth and C. Schymura},
    booktitle = {Konferenz zur Verarbeitung natürlicher Sprache/Conference on Natural Language Processing (KONVENS)},
    title = {Data {S}cience {K}itchen at {G}erm{E}val 2021: {A} Fine Selection of Hand-Picked Features, Delivered Fresh from the Oven},
    year = {2021},
    pages = {88-94}
}
```

## 🧑‍⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
