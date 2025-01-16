# FRU-Adapter: Frame recalibration unit adapter for dynamic facial expression recognition

> Seoul National University of Science and Technology & HCIR Lab<br>
## 📰 News
**[2025.01.16]** We upload the code of FRU-Adapter <br>

## ✨ Overview

Dynamic facial expression recognition (DFER) is one of the important challenges in com
puter vision, as it plays a crucial role in human-computer interaction. Recently, adapter-based
 approaches have been introduced to the DFER and they have achieved remarkable success. However,
 the adapters still suffer from the following problems: overlooking irrelevant frames and interference
 with pre-trained information. In this paper, we propose a frame recalibration unit adapter (FRU
Adapter), which combines the strengths of frame recalibration unit (FRU) and temporal self-attention
 (T-SA) to address the aforementioned issues. 
 
<p align="center">
  <img src="figs/FRU-Adapter" width=50%> <br>
  Overall architecture of FRU-Adapter.
</p>

## 🚀 Main Results

### ✨ Dynamic Facial Expression Recognition

![Result_on_DFEW, FERV39k, MAFW dataset](figs/Result_on_DFEW.png)

## 🔨 Installation

Main prerequisites:

* `Python 3.8`
* `PyTorch 1.7.1 (cuda 10.2)`
* `timm==0.4.12`
* `einops==0.6.1`
* `decord==0.6.0`
* `scikit-learn=1.1.3`
* `scipy=1.10.1`
* `pandas==1.5.3`
* `numpy=1.23.4`
* `opencv-python=4.7.0.72`
* `tensorboardX=2.6.1`

If some are missing, please refer to [environment.yml](environment.yml) for more details.


## ➡️ Data Preparation

Please follow the files (e.g., [dfew.py](preprocess/dfew.py)) in [preprocess](preprocess) for data preparation.

Specifically, you need to enerate annotations for dataloader ("<path_to_video> <video_class>" in annotations). 
The annotation usually includes `train.csv`, `val.csv` and `test.csv`. The format of `*.csv` file is like:

```
dataset_root/video_1  label_1
dataset_root/video_2  label_2
dataset_root/video_3  label_3
...
dataset_root/video_N  label_N
```

An example of [train.csv](saved/data/dfew/org/split01/train.csv) of DFEW fold1 (fd1) is shown as follows:

```
/mnt/data1/brain/AC/Dataset/DFEW/Clip/jpg_256/02522 5
/mnt/data1/brain/AC/Dataset/DFEW/Clip/jpg_256/02536 5
/mnt/data1/brain/AC/Dataset/DFEW/Clip/jpg_256/02578 6
```

Note that, `label` for the pre-training dataset (i.e., VoxCeleb2) is dummy label, you can simply use `0` (see [voxceleb2.py](preprocess/voxceleb2.py)).

## Fine-tune with pre-trained weights
1、 Download the pre-trained weights from [baidu drive](https://pan.baidu.com/s/1J5eCnTn_Wpn0raZTIUCfgw?pwd=dji4) or [google drive](https://drive.google.com/file/d/1Y9zz8z_LwUi-tSFBAwDPZkVoyY6mhZlu/view?usp=drive_link) or [onedrive](https://mailhfuteducn-my.sharepoint.com/:f:/g/personal/2022111029_mail_hfut_edu_cn/EgKQNq8Y2chKl2TSoYf_OA0BQpCwx-FDw2ksPaMxBntZ8A), and move it to the ckpts directory.

2、 Run the following command to fine-tune the model on the target dataset.
```bash
conda create -n FRU_Adapter python=3.9
conda activate FRU_Adapter
pip install -r requirements.txt
bash run.sh
```

## 📋 Reported Results and Fine-tuned Weights
The fine-tuned checkpoints can be downloaded from [here](https://pan.baidu.com/s/1Xz5j8QW32x7L0bnTEorUbA?pwd=5drk).
<table border="1" cellspacing="0" cellpadding="5">
    <tr>
        <th rowspan="2">Datasets</th>
        <th colspan="2">16-frame uniformsampling</th>
        <th colspan="2">2-clip average</th>
    </tr>
    <tr>
        <th>UAR</th>
        <th>WAR</th>
        <th>UAR</th>
        <th>WAR</th>
    </tr>
    <tr><td colspan="5" style="text-align: center;">FERV39K</td></tr>
    <tr>
        <td>FERV39K</td>
        <td>38.65</td>
        <td>50.12</td>
        <td>41.08</td>
        <td>52.70</td>
    </tr>
    <tr><td colspan="5" style="text-align: center;">DFEW</td></tr>
    <tr>
        <td>DFEW01</td>
        <td>66.12</td>
        <td>77.22</td>
        <td>64.28</td>
        <td>76.89</td>
    </tr>
    <tr>
        <td>DFEW02</td>
        <td>63.12</td>
        <td>75.13</td>
        <td>63.85</td>
        <td>74.88</td>
    </tr>
    <tr>
        <td>DFEW03</td>
        <td>64.79</td>
        <td>76.84</td>
        <td>65.78</td>
        <td>76.37</td>
    </tr>
    <tr>
        <td>DFEW04</td>
        <td>66.14</td>
        <td>77.05</td>
        <td>66.39</td>
        <td>78.20</td>
    </tr>
    <tr>
        <td>DFEW05</td>
        <td>69.91</td>
        <td>78.55</td>
        <td>69.10</td>
        <td>78.55</td>
    </tr>
    <tr>
        <td>DFEW</td>
        <td>66.02</td>
        <td>76.96</td>
        <td>65.88</td>
        <td>76.98</td>
    </tr>
    <tr><td colspan="5" style="text-align: center;">MAFW</td></tr>
    <tr>
        <td>MAFW01</td>
        <td>34.70</td>
        <td>48.29</td>
        <td>38.42</td>
        <td>51.82</td>
    </tr>
    <tr>
        <td>MAFW02</td>
        <td>41.66</td>
        <td>55.58</td>
        <td>42.13</td>
        <td>56.18</td>
    </tr>
    <tr>
        <td>MAFW03</td>
        <td>49.21</td>
        <td>62.41</td>
        <td>48.40</td>
        <td>62.25</td>
    </tr>
    <tr>
        <td>MAFW04</td>
        <td>46.58</td>
        <td>64.08</td>
        <td>49.36</td>
        <td>65.23</td>
    </tr>
    <tr>
        <td>MAFW05</td>
        <td>41.89</td>
        <td>58.77</td>
        <td>44.40</td>
        <td>61.17</td>
    </tr>
    <tr>
        <td>MAFW</td>
        <td>42.80</td>
        <td>57.83</td>
        <td>44.54</td>
        <td>59.33</td>
    </tr>
</table>

## ☎️ Contact 

If you have any questions, please feel free to reach me out at `gblader@naver.com`.

## 👍 Acknowledgements

This research was supported by the MSIT(Ministry of Science and ICT), Korea, under 
the ITRC(Information Technology Research Center) support program (IITP-2025-RS-2022-00156295) 
supervised by the IITP(Institute for Information Communications Technology Planning Evaluation).

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:



