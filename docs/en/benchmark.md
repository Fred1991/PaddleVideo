# Benchmark

This document provides the benchmark of the prediction time of PaddleVideo series models on various platforms.

---

## Index

- [1. Video Classification Models](#1)
    - [1.1 Data](#11)
    - [1.2 Environment](#12)
    - [1.3 Results](#13)
        - [1.3.1 Overview of GPU Inference Speeds](#131)
        - [1.3.2 Overview of CPU Inference Speeds](#132)
    - [1.4 Testing Methods](#14)
        - [1.4.1 Single Model Test](#141)
        - [1.4.2 Batch Test](#141)



## 1. Video Classification Models

### 1.1 Data

We randomly selected 100 entries from the Kinetics-400 dataset for benchmark time testing. The test data can be accessed by clicking on the provided [link](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/time-test.tar).

The directory of files after decompression:
```txt
time-test
├── data       
└── file.list  
```

The video properties are as follows:

```txt
mean video time:  9.67s
mean video width:  373
mean video height:  256
mean fps:  25
```

### 1.2 Environment

Hardware：

- CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- GPU: Tesla V100 16G

Software：
- Python 3.7
- PaddlePaddle 2.3.1
- CUDA 10.2
- CUDNN 8.1.1
- Packages in [requirement.txt](../../requirements.txt)

### 1.3 Results

#### 1.3.1 Overview of GPU Inference Speeds

The performance data of each model is sorted by the total prediction time, and the results are as follows:

| Model | Backbone | Config | Acc (%) | Prep Time (ms) | Inference Time (ms) | Total Time (ms) |
| :---- | :---- | :----: |:----: |:----: |:----: |:----: |
| $LiteTSM^-$ | MobileNetV2 | [pptsm_mv2_k400_videos_uniform.yaml](../../configs/recognition/pptsm/pptsm_mv2_k400_videos_uniform.yaml) | 68.09 | 51.5 | 3.31 | 54.81 |
| $LiteTSM^-$ | MobileNetV3 | [pptsm_mv3_k400_frames_uniform.yaml](../../configs/recognition/pptsm/pptsm_mv3_k400_frames_uniform.yaml) | 69.84 | 51 | 4.34 | 55.34 |
| **$\textbf{LiteTSM}$** | PP-LCNet_v2.8f |	[pptsm_lcnet_k400_8frames_uniform.yaml](../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_8frames_uniform.yaml) | **72.45**| 55.31 | 4.37 | **59.68** |
| TSM | R50 | [tsm_k400_frames.yaml](../../configs/recognition/tsm/tsm_k400_frames.yaml) | 71.06 | 52.02 | 9.87 | 61.89 |
|**$LiteTSM^-$**	| R50 |	[pptsm_k400_frames_uniform.yaml](../../configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml) | **75.11** | 51.84 | 11.26 | **63.1** |
|$LiteTSM^-$	| R101 | [pptsm_k400_frames_dense_r101.yaml](../../configs/recognition/pptsm/pptsm_k400_frames_dense_r101.yaml) | 76.35| 52.1 | 17.91 | 70.01 |
| $\textbf{LiteTSM}$ | PP-LCNet_v2.16f |	[pptsm_lcnet_k400_16frames_uniform.yaml](../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform.yaml) | 74.38 |  69.4 | 7.55 | 76.95 |
| SlowFast | 4*16 |	[slowfast.yaml](../../configs/recognition/slowfast/slowfast.yaml) | 74.35 | 99.27 | 27.4 | 126.67 |
| *VideoSwin | B | [videoswin_k400_videos.yaml](../../configs/recognition/videoswin/videoswin_k400_videos.yaml) | 82.4 | 95.65 | 117.22 | 212.88 |
| MoViNet | A0 | [movinet_k400_frame.yaml](../../configs/recognition/movinet/movinet_k400_frame.yaml) | 66.62 | 150.36 | 47.24 | 197.60 |
| *TimeSformer+ | base | [pptimesformer_k400_videos.yaml](../../configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml) | 78.87 | 299.48 | 133.41 | 432.90 |
| *TimeSformer |	base |	[timesformer_k400_videos.yaml](../../configs/recognition/timesformer/timesformer_k400_videos.yaml) | 77.29 | 301.54 | 136.12 | 437.67 |
| TSN | R50	| [tsn_k400_frames.yaml](../../configs/recognition/tsn/tsn_k400_frames.yaml) | 69.81 | 794.30 | 168.70 | 963.00 |
| LiteTSN | R50 | [pptsn_k400_frames.yaml](../../configs/recognition/pptsn/pptsn_k400_frames.yaml) | 75.06 | 837.75 | 175.12 | 1012.87 |

* indicates that the model has not been accelerated with tensorRT for inference speedup.

- When predicting with TSN, TenCrop is used, which is more time-consuming than CenterCrop used by TSM. If TSN uses CenterCrop, the speed is slightly better than TSM, but the accuracy will be 3.5 points lower.

#### 1.3.2 Overview of CPU Inference Speeds

The performance data of each model is sorted by the total prediction time, and the results are as follows:

| Model | Backbone | Config | Acc (%) | Prep Time (ms) | Inference Time (ms) | Total Time (ms) |
| :---- | :---- | :----: |:----: |:----: |:----: |:----: |
| $LiteTSM^-$ | MobileNetV2 | [pptsm_mv2_k400_videos_uniform.yaml](../../configs/recognition/pptsm/pptsm_mv2_k400_videos_uniform.yaml) | 68.09 | 52.62 | 137.03 | 189.65 |
| $LiteTSM^-$ | MobileNetV3 | [pptsm_mv3_k400_frames_uniform.yaml](../../configs/recognition/pptsm/pptsm_mv3_k400_frames_uniform.yaml) | 69.84| 53.44 | 139.13 | 192.58 |
| **$\textbf{LiteTSM}$** | PP-LCNet_v2.8f |	[pptsm_lcnet_k400_8frames_uniform.yaml](../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_8frames_uniform.yaml) | **72.45**| 53.37 | 189.62 | **242.99** |
| **$\textbf{LiteTSM}$** | PP-LCNet_v2.16f |	[pptsm_lcnet_k400_16frames_uniform.yaml](../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform.yaml) | **74.38**|  68.07 | 388.64 | **456.71** |
| SlowFast | 4*16 |	[slowfast.yaml](../../configs/recognition/slowfast/slowfast.yaml) | 74.35 | 110.04 | 1201.36 | 1311.41 |
| TSM | R50 | [tsm_k400_frames.yaml](../../configs/recognition/tsm/tsm_k400_frames.yaml) | 71.06 | 52.47 | 1302.49 | 1354.96 |
|$LiteTSM^-$	| R50 |	[pptsm_k400_frames_uniform.yaml](../../configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml) | 75.11 | 52.26  | 1354.21 | 1406.48 |
|*MoViNet | A0 | [movinet_k400_frame.yaml](../../configs/recognition/movinet/movinet_k400_frame.yaml) | 66.62 | 148.30 |	1290.46 | 1438.76 |
|$LiteTSM^-$	| R101 | [pptsm_k400_frames_dense_r101.yaml](../../configs/recognition/pptsm/pptsm_k400_frames_dense_r101.yaml) | 76.35| 52.50 | 2236.94 | 2289.45 |
| TimeSformer+ | base | [pptimesformer_k400_videos.yaml](../../configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml) | 78.87 | 294.89	| 13426.53 | 13721.43 |
| TimeSformer |	base |	[timesformer_k400_videos.yaml](../../configs/recognition/timesformer/timesformer_k400_videos.yaml) | 77.29 | 297.33 |	14034.77 |	14332.11 |
| TSN | R50	| [tsn_k400_frames.yaml](../../configs/recognition/tsn/tsn_k400_frames.yaml) | 69.81 | 860.41 | 18359.26 | 19219.68 |
| LiteTSN | R50 | [pptsn_k400_frames.yaml](../../configs/recognition/pptsn/pptsn_k400_frames.yaml) | 75.06 | 835.86 | 19778.60 | 20614.46 |
| *VideoSwin | B | [videoswin_k400_videos.yaml](../../configs/recognition/videoswin/videoswin_k400_videos.yaml) | 82.4 | 76.21 | 32983.49 | 33059.70 |


* indicates that the model has not been accelerated with mkldnn for inference speedup.


### 1.4 Testing Methods

Before testing, you need to install the related dependencies in [requirements.txt](../../requirements.txt), and also install `AutoLog` to record the computation time. Use the following command to install:
```bash
python3.7 -m pip install --upgrade pip
pip3.7 install --upgrade -r requirements.txt
python3.7 -m pip install git+https://github.com/LDOUBLEV/AutoLog
```

#### 1.4.1 Single Model Test

Using the PP-TSM model as an example, please refer to the [PP-TSM文档](./model_zoo/recognition/pp-tsm.md) document to export the inference model. Then, use the following command to perform speed testing:

```python
python3.7 tools/predict.py --input_file time-test/file.list \
                          --time_test_file=True \
                          --config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
                          --model_file inference/ppTSM/ppTSM.pdmodel \
                          --params_file inference/ppTSM/ppTSM.pdiparams \
                          --use_gpu=False \
                          --use_tensorrt=False \
                          --enable_mkldnn=True \
                          --enable_benchmark=True \
                          --disable_glog True
```

- The meaning of each parameter is as follows：

```txt
input_file:     Specify the test file/file list, using the test data provided in section 1.1 as an example.
time_test_file: whether to do the time test，please set it to True.
config:         Specify the model configuration file.
model_file:     Specify the path to the inference file of the pdmodel.
params_file:    Specify the path to the inference file of the pdiparams.
use_gpu:        whether to use GPU to do the inference, False means using CPU to test.
use_tensorrt:   whether to luanch TensorRT to test.
enable_mkldnn:  whether to use mkldnn，default value is True.
disable_glog:   whehter to disable inference log，please set it to True.
```

- During testing, GPU inference is performed using the FP32+TensorRT configuration, while CPU acceleration is achieved using mkldnn. After running the inference 100 times and excluding the warmup time of the first 3 runs, the average inference time is obtained.

#### 1.4.2 Batch Tests

The following batch testing script can easily reproduce performance results:

- 1. Download the pretrained models:

```bash
mkdir ckpt
cd ckpt
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform_distill.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSM_k400_uniform_distill_r101.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_mv2_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_mv3_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.3/PPTSMv2_k400_16f_dml.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400_8.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTimeSformer_k400_8f_distill.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TSN_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TimeSformer_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.3/MoViNetA0_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_k400.pdparams
```

- 2. Prepare configuration parameter lists for each model `model.list`

```txt
PP-TSM_R50      configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml        ckpt/ppTSM_k400_uniform_distill.pdparams ppTSM
PP-TSM_R101     configs/recognition/pptsm/pptsm_k400_frames_dense_r101.yaml     ckpt/ppTSM_k400_uniform_distill_r101.pdparams ppTSM
PP-TSM_MobileNetV2      configs/recognition/pptsm/pptsm_mv2_k400_videos_uniform.yaml    ckpt/ppTSM_mv2_k400.pdparams ppTSM
PP-TSM_MobileNetV3      configs/recognition/pptsm/pptsm_mv3_k400_frames_uniform.yaml    ckpt/ppTSM_mv3_k400.pdparams ppTSM
PP-TSMv2_PP-LCNet_v2    configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform_dml_distillation.yaml      ckpt/PPTSMv2_k400_16f_dml.pdparams ppTSMv2
PP-TSN_R50      configs/recognition/pptsn/pptsn_k400_frames.yaml        ckpt/ppTSN_k400_8.pdparams ppTSN
PP-TimeSformer_base     configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml        ckpt/ppTimeSformer_k400_8f_distill.pdparams ppTimeSformer
TSM_R50 configs/recognition/tsm/tsm_k400_frames.yaml    ckpt/TSM_k400.pdparams TSM
TSN_R50 configs/recognition/tsn/tsn_k400_frames.yaml    ckpt/TSN_k400.pdparams TSN
TimeSformer_base        configs/recognition/timesformer/timesformer_k400_videos.yaml    ckpt/TimeSformer_k400.pdparams TimeSformer
SlowFast_416    configs/recognition/slowfast/slowfast.yaml      ckpt/SlowFast.pdparams SlowFast
MoViNet_A0      configs/recognition/movinet/movinet_k400_frame.yaml     ckpt/MoViNetA0_k400.pdparams MoViNet
VideoSwin_B     configs/recognition/videoswin/videoswin_k400_videos.yaml        ckpt/VideoSwin_k400.pdparams VideoSwin
```

- 3. Bulk export model, pass in model.list file at execution time.

```bash
file=$1

while read line
do
    arr=($line)
    ModelName=${arr[0]}
    ConfigFile=${arr[1]}
    ParamsPath=${arr[2]}
    echo $ModelName

    python3.7 tools/export_model.py -c $ConfigFile \
                                    -p $ParamsPath \
                                    -o inference/$ModelName
done <$file
```

- 4. Test time, pass in the model.list file at runtime.

```bash
file=$1

while read line
do
    arr=($line)
    ModelName=${arr[0]}
    ConfigFile=${arr[1]}
    ParamsPath=${arr[2]}
    Model=${arr[3]}

    python3.7 tools/predict.py --input_file ../../time-test/file.list \
                            --time_test_file=True \
                            --config $ConfigFile \
                            --model_file inference/$ModelName/$Model.pdmodel \
                            --params_file inference/$ModelName/$Model.pdiparams \
                            --use_gpu=False \
                            --use_tensorrt=False \
                            --enable_mkldnn=False \
                            --enable_benchmark=True \
                            --disable_glog True
    echo =====$ModelName END====
done <$file
```

---


