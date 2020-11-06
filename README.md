# Zalo AI Challenge 2020 - Voice Verification


## Description

Voice verification is the process of verifying whether two given utterances belong to a single speaker.
In this challenge, we want you to build a speaker text-independent verification model for Vietnamese voices.

- Input: a pair of utterances. Each utterance's length is from 0.8s-11s.
- Output: Prediction of whether these utterances belong to a single speaker.


## Data

**Training dataset**: The training set should be used to build your models.
For the training set, we provide 400 speakers corresponding to 400 folders.
The folder name has the following format {id-gender-age}.

**Public-test** (public-test.csv): The test set should be used to see how well your model performs on unseen data.
For the public test set, we have 2789 files that make up 50,000 pairs of utterances for evaluation.

- Public-test.csv:  This file contains all test cases of the public test set. Each test case includes a pair of utterances for comparison:
    - 'audio_1': name of the first audio file
    - 'audio_2': name of the second audio file
- Sample_submission.csv: The header is audio_1, audio_2 and label. The first two columns have the same order as in the file pubic-test.csv.
    - label = 1 when these utterances are the same speaker
    - label = 0 when these utterances are not the same speaker

- Download npy format dataset at [here](https://drive.google.com/file/d/1sWwITvIoUiyZa44yuKPUygPvmRGOb1xL/view?usp=sharing)

- Download wavs format dataset at [here](https://drive.google.com/file/d/10abB9_1QRf-5_1CWPnQtu4zsH5Qmy-dA/view?usp=sharing)

- Dowload by command line
```
pip3 install gdown # or pip
gdown https://drive.google.com/uc?id=10abB9_1QRf-5_1CWPnQtu4zsH5Qmy-dA # wav sample rate = 48000
gdown https://drive.google.com/uc?id=1sWwITvIoUiyZa44yuKPUygPvmRGOb1xL # npy sample rate = 22050
unzip wav.zip -d dataset/; rm wav.zip
unzip npy.zip -d dataset/; rm npy.zip
```

## Enviroments
- We use `tensorflow-gpu` 2.3.1 that is compatible with `CUDA` 10.1. [Install CUDA](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)

- How to 

- How to install Nvidia driver correctly?

    + Remove old `CUDA` completely from `Ubuntu`?

    ```
    sudo apt-get purge nvidia*
    sudo apt-get purge cuda*
    sudo apt-get autoremove
    sudo apt-get autoclean
    sudo rm /etc/apt/sources.list.d/cuda*
    sudo rm -rf /usr/local/cuda*
    ```

    + Now it is good time to install correct NVIDIA drivers. For example from [here](https://www.nvidia.com/Download/index.aspx). P/S

    ```
    # cuda-runtime-10-1 : Depends: cuda-drivers (>= 418.39)
    # do not run this command to install driver! this has error all the time!
    # sudo ubuntu-drivers autoinstall
    ```

- How to install `CUDA` 10.1 on `Ubuntu`?

```
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.105-418.39/7fa2af80.pub
sudo apt update
sudo apt install cuda-runtime-10-1 cuda-10-1 cuda
```

## Author
[KhoiDD](https://github.com/mazino2d), [ThucTH](https://github.com/thucth-qt)