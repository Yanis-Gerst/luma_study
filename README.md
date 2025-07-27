# LUMA+DBF: Learning from Uncertain and Multimodal Data

## Getting Started

First, compile the dataset as you normally would for LUMA to create the database. Instructions for LUMA installation are provided below.

Second, clone the OctoPy repository:

```bash
git clone octopy
```

Then, create a Python virtual environment:

```bash
python -m venv .venv
```

Install OctoPy's dependencies:

```bash
pip install -r octopy/requirements.txt
```

Install LUMA's dependencies:

```bash
pip install -r requirements.txt
```

You can now run the code. Apologies for the disorganization.

## Run the Code

There are two files to run:

The first file is `run_baselines.py`, which is used for training and testing the model on the LUMA dataset. You can run it with the following command:

This file has many parameters due to extensive debugging and testing. The main parameters are:

  * `--id`: The ID of the model, used for saving.
  * `--model`: The model to use. Options include `image`, `text`, `audio`, `multimodal`, or `multimodalWithTwo`.
  * `--flambda`: Specific to DBF.
  * `--mode`: Use `train` to train the model and save its weights, or `test` to load weights and evaluate the model, saving the results.

Example command:

```bash
python run_baselines.py --id 1 --model audio --mode train
```

To run a multimodal model, you need pre-trained unimodal models. The code will load these based on their IDs. For example, if your multimodal model has an ID of `1`, it will load weights from `audio_1` and `image_1`.


For creating the plots, you have to run `plots.py` file. In that file you have a name variable that store the path to the results json file you want to plot. 

### Installation
Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/bezirganyan/LUMA.git 
cd LUMA
```
Install and activate the conda enviroment
```bash
conda env create -f environment.yml
conda activate luma_env
```

Make sure you have git-lfs installed (https://git-lfs.com), it will be automatically installed by conda if you did previous steps. Then perform:
```
git lfs install
```
Download the dataset under the `data` folder (you can also choose other folder names, and updated config files, `data` folder is the default in the default configurations)
```bash
git clone https://huggingface.co/datasets/bezirganyan/LUMA data
```

### Usage
The provided Python tool allows compiling different versions of the dataset with various amounts and types of uncertainties.

To compile the dataset with specified uncertainties, create or edit the configuration file similar to the files in `cfg` directory, and run:
```
python compile_dataset.py -c <your_yaml_config_file>
```

### Usage in Deep Learning models
After compiling the dataset, you can use the `LUMADataset` class from the `dataset.py` file. Example of the usage can be found in `run_baselines.py` file.

### Unprocessed & Unaigned data
If you want to get all the data (without sampling or noise) without alignment (to perform your own alignment, or use the data without alignment for other tasks) you can run the following command:

```
python get_unprocessed_data.py
```

If you use the dataset, please cite our paper with:
```
@inproceedings{luma_dataset2025,
  title={LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data}, 
  author={Grigor Bezirganyan and Sana Sellami and Laure Berti-Équille and Sébastien Fournier},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```
