# Context-Aware Adversarial Attacks
### [Paper](http://arxiv.org/abs/2112.03223) | [Code](https://github.com/SUPERustam/context-aware-attacks) | [Results of Sequential Attacks](evaluate/evaluate.txt) | [Results of Transfer Attacks](evaluate/evaluate_bb.txt) 

[Reporduction] Pytorch implementation of *Context-Aware Transfer Attacks for Object Detection* in AAAI 2022.

>_The original repo have some implicit details/small bugs which I fix/show in this repo_

[Context-Aware Transfer Attacks for Object Detection](http://arxiv.org/abs/2112.03223)  
 [Zikui Cai](https://zikuicai.github.io/), Xinxin Xie, Shasha Li, Mingjun Yin, Chengyu Song,Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury,
 [M. Salman Asif](https://intra.ece.ucr.edu/~sasif/)<br>
 UC Riverside 

Blackbox transfer attacks for image classifiers have been extensively studied in recent years. In contrast, little progress has been made on transfer attacks for object detectors. Object detectors take a holistic view of the image and the detection of one object (or lack thereof) often depends on other objects in the scene. This makes such detectors inherently context-aware and adversarial attacks in this space are more challenging than those targeting image classifiers. In this paper, we present a new approach to generate context-aware attacks for object detectors. We show that by using co-occurrence of objects and their relative locations and sizes as context information, we can successfully generate targeted mis-categorization attacks that achieve higher transfer success rates on blackbox object detectors than the state-of-the-art. We test our approach on a variety of object detectors with images from PASCAL VOC and MS COCO datasets and demonstrate up to 20 percentage points improvement in performance compared to the other state-of-the-art methods.

<img src='doc/framework.png'>


## Environment
See `requirements.txt` (_update corect installtion instructures_), some key dependencies are:

* python==3.7
* torch==1.7.0 
* mmcv-full==1.3.3

Install mmcv-full https://github.com/open-mmlab/mmcv.

```
pip install mmcv-full==1.3.3 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
# depending on your cuda version (cuda 11 the newest for 1.3.3)
```

## Datasets
Get VOC2007 and COCO2017 datasets under `/data` folder.
```
cd data
bash get_voc.sh
bash get_coco.sh
```

## Object Detection Models
Get mmdetection code repo and download pretrained models (_also fixed paths_)
```
cd detectors
git clone https://github.com/zikuicai/mmdetection
# This will download mmdetection package to detectors/mmdetection/

python mmdet_model_info.py
# This will download checkpoin files into detectors/mmdetection/checkpoints
```

## Attacks and Evaluation
Run sequential attack. 
>For reproducting original results, run every dataset in different folder (aka `root`) for futher exploring. Also run at every perturbation level: 10, 20, 30
```sh
cd attacks

# for each experiment run like this. Every sequential_attack use ~7GB of VRAM in GPU
python run_sequential_attack.py --eps 10 --root result_COCO --dataset coco
... --eps 20
... --eps 30
python run_sequential_attack.py --eps 10 --root result_VOC --dataset voc
... --eps 20
... --eps 30
```

Calculate fooling rate.
```sh
cd evaluate
# for each experiment run like this.
python get_fooling_rate.py --eps 10 --root result_VOC
...
...
```

Run transfer attacks on different blackbox models.
```sh
cd attacks

# for each experiment run like this. Every run_transfer_attack use ~3GB of VRAM in GPU
python run_transfer_attack.py --eps 10 --root result_COCO --dataset coco
... --eps 20
... --eps 30
```

Calculate fooling rate again on blackbox results.
```sh
cd evaluate
# for each experiment run like this.
python get_fooling_rate.py --eps 10 --root result_VOC --bb
...
...
```

_if you accidently run several times same script run_sequential_attack.py or run_transfer_attack.py try run my custom [script](attacks/deduplicate_clean_attacks.py) to clean duplicates and wrong cache files_
```sh
cd attacks
# for example
python deduplicate_clean_attacks.py --eps 10 --root result_VOC --dataset voc
```
CLI interface of [deduplicate_clean_attacks.py](attacks/deduplicate_clean_attacks.py)
```py
    parser = argparse.ArgumentParser(
        description="Deduplicate attack plans and remove wrong attack plans."
    )

    parser.add_argument(
        "--eps", nargs="?", default=30, help="perturbation level: 10,20,30,40,50"
    )
    parser.add_argument(
        "--root", nargs="?", default="result", help="the folder name of result"
    )
    parser.add_argument("-bb", action="store_true", help="use bb txt file")
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="voc",
        help="model dataset 'voc' or 'coco'. This will change txt file name",
    )
```

## Overview of Code Structure
- data
    - script to download datasets VOC and COCO
    - indices of images used in our experiments   
- detectors
    - packages for object detectors
    - script to download the pretrained model weights
    - util and visualization functions for mmdetection models
- context
    - co-occurrence matrix
    - distance matrix
    - size matrix
- attacks
    - code to attack the detectors
    - code to transfer attack other blackbox detectors
    - _code to duplicate attack plans_
- evaluate
    - code to calculate the fooling rate of whitebox and blackbox attacks


## Citation
```
@inproceedings{cai2021context,
  title={Context-Aware Transfer Attacks for Object Detection},
  author={Cai, Zikui and Xie, Xinxin and Li, Shasha and Yin, Mingjun and Song, Chengyu and Krishnamurthy, Srikanth V and Roy-Chowdhury, Amit K and Asif, M Salman},
  year={2022},
  booktitle={AAAI}
}
```
