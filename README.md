# Post-hoc Probabilistic Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2412.06014-b31b1b.svg)](https://arxiv.org/abs/2412.06014)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://aaltoml.github.io/BayesVLM/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/collections/aalto-ml/bayesvlm)


![image](pipeline.png)

# Setup Instructions
1. Ensure you have Python version `>= 3.11` installed.
2. Install the required packages by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Set `DATA_BASE_DIR` in your `.env` file. You can use the structure from the `.env.example` file.
   ```
   DATA_BASE_DIR=/path/to/datasets
   ```
4. Add the project root directory to the `PYTHONPATH` environment variable.
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/project/root
   ```
4. (Optional) If you use a M1 Mac with `mps` support you'll need to set the following environment variable:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```
# Running the Code
To run the hessian estimation code, use the following command:
```bash
python scripts/hessian_estimation.py
```

To run the code for zero-shot experiments, use the following command:
```bash
python scripts/zeroshot.py
```

To run the code for the active-learning experiments, use the following command:
```bash
python scripts/activelearning.py
```

Note that each of those commands has additional arguments that allow the adjustment of the Hessian estimation and zero-shot/active learning experiments.

# Hessians
The precomputed Hessians for the models used in the paper are available in the `hessians/` folder. You can select a specific hessian by setting `--hessian_dir` in the provided scripts.

# Notebooks
A notebook stepping through the zero-shot code is available in `notebooks/zeroshot.ipynb`.

# Data Setup
The data is stored in the `DATA_BASE_DIR` folder and is structured as follows:
```bash
DATA_BASE_DIR/
├── cifar10/
├── cifar100/
├── eurosat/
├── flowers102/
├── food101/
├── homeoffice/
├── imagenet_r/
├── imagenet_val_wds/
├── laion400m/
├── sun397/
├── ucf101/
```
Please set the `DATA_BASE_DIR` environment variable accordingly.

### CIFAR-10
The `CIFAR-10` dataset is automatically downloaded by the huggingface `datasets` library.

### CIFAR-100
The `CIFAR-100` dataset is automatically downloaded by the huggingface `datasets` library.

### EuroSAT
From https://github.com/vishaal27/SuS-X/blob/main/data/DATA.md
- Create a folder named `eurosat/` under `DATA_BASE_DIR`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `DATA_BASE_DIR/eurosat/`.
- Download `split_zhou_EuroSAT.json` from [here](https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing) and put it under `DATA_BASE_DIR/eurosat`.

The directory structure should look like
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### Flowers102
The `Flowers102` dataset is automatically downloaded by the `torchvision` library.

### Food101
The `Food101` dataset is automatically downloaded by the `torchvision` library.

### HomeOffice
Download the dataset from https://www.hemanthdv.org/officeHomeDataset.html and extract it to `DATA_BASE_DIR/homeoffice/`.

The directory structure should look like
```
homeoffice/
|–– Art/
|–– Clipart/
|–– Product/
|–– Real World/
|–– ImageInfo.csv
|–– imagelist.txt
```

### Stanford Cars
Follow the instructions https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616 to download the dataset and extract it to `DATA_BASE_DIR/stanford_cars/`.

### DTD
The `DTD` dataset is automatically downloaded by the `torchvision` library.

### Imagenet Web-Dataset (val)
We supply the script `scripts/download_imagenet.py` to download all validation tar files for the ImageNet dataset from the Hugging Face Datasets Hub.
After running the script, the directory structure should look like
```
imagenet_val_wds/
|–– imagenet1k-validation-00.tar
|–– imagenet1k-validation-01.tar
|–– ...
|–– imagenet1k-validation-63.tar
```

### Imagenet-R
Download the dataset from https://github.com/hendrycks/imagenet-r and extract it to `./data/imagenet-r/`.

### Laion400M
The `laion400M` dataset can be downloaded using the [img2dataset](https://github.com/rom1504/img2dataset) tool. The instructions for the `laion400m` dataset are available [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md).
Before running the `img2dataset` script, we removed all data points marked as `NSFW` in the metadata.

### SUN397
- Create a folder named  `sun397/` under `./data`.
- Download the images http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
- Download the partitions https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
- Extract these files under `./data/sun397/`.
- Download `split_zhou_SUN397.json` from this [link](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing) and put it under `./data/sun397`.

The directory structure should look like
```
sun397/
|–– SUN397/
|–– split_zhou_SUN397.json
|–– ... # a bunch of .txt files
```

### UCF101
- Create a folder named `ucf101/` under `./data`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `./data/ucf101/`. This zip file contains the extracted middle video frames.
- Download `split_zhou_UCF101.json` from this [link](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing) and put it under `./data/ucf101`.

The directory structure should look like
```
ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```


## Citation

```bibtex
@inproceedings{baumann2026bayesvlm,
  title     = {Post-hoc Probabilistic Vision-Language Models},
  author    = {Baumann, Anton and Li, Rui and Klasson, Marcus and Mentu, Santeri and Karthik, Shyamgopal and Akata, Zeynep and Solin, Arno and Trapp, Martin},
  booktitle = {International Conference on Learning Representations {(ICLR)}},
  year      = {2026},
}
```

## License
This software is provided under the MIT license.
