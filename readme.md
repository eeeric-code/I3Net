# I3Net: Inter-Intra-slice Interpolation Network for Medical Slice Synthesis (TMI 2024)
Haofei Song, Xintian Mao, Jing Yu, Qingli Li, Yan Wang

**Paper:** <https://doi.org/10.1109/TMI.2024.3394033>.



## Requirements
This repository is based on PyTorch 1.12.0, CUDA 11.3 and Python 3.8.13. 

## Data Preparation
**Data obtain:** 
- **CT:** From [Medical Decathlon Challenge (MSD)](http://medicaldecathlon.com/), specifically from the [Liver](https://drive.google.com/file/d/1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu/view), [Colon](https://drive.google.com/file/d/1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y/view), and [Hepatic Vessel](https://drive.google.com/file/d/1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS/view) dataset. Data split using [test_set.pt](data_prepare/test_set.pt) from [SAINT](https://github.com/cpeng93/SAINT).

- **MR:** From [IXI](https://brain-development.org/ixi-dataset/), collected at Hammersmith Hospital.

**Data process:** 
- Split trainset and testset using [step1_split.py](data_prepare/step1_split.py).
- Save each slice using [step2_save_slice.py](data_prepare/step2_save_slice.py).

## Quick run

To test a model: 
```
bash test.sh
```

To train a model: 
```
bash train.sh
```
## Citation
```
```


## Contact
If you have any question, please contact <hfsong@stu.ecnu.edu.cn>
