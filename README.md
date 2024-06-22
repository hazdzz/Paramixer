# Paramixer

## About
The PyTorch implementation of Paramixer from the paper [*Paramixer: Parameterizing Mixing Links in Sparse Factors Works Better than Dot-Product Self-Attention*](<https://arxiv.org/abs/2204.10670>).

## Citation
```
@inproceedings{9878955,
  title     = {Paramixer: Parameterizing Mixing Links in Sparse Factors Works Better than Dot-Product Self-Attention}, 
  author    = {Yu, Tong and Khalitov, Ruslan and Cheng, Lei and Yang, Zhirong},
  booktitle = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  year      = {2022},
  pages     = {681--690}
}
```

## Datasets
1. LRA: https://mega.nz/file/sdcU3RKR#Skl5HomJJldPBqI7vfLlSAX8VA0XKWiQSPX1E09dwbk

## Training Steps
1. Create a data folder:
```console
mkdir data
```

2. Download the dataset compressed archive
```console
wget $URL
```

3. Decompress the dataset compressed archive and put the contents into the data folder
```console
unzip $dataset.zip
mv $datast ./data/$datast
```

4. Run the main file
```console
python $dataset_main.py --task="$task"
```

## Requirements
To install requirements:
```console
pip3 install -r requirements.txt
