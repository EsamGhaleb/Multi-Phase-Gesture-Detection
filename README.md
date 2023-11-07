# Co-Speech Gesture Detection through Multi-Phase Sequence Labeling
---

This repository contains the code of the WACV2024 paper titled "Co-Speech Gesture Detection through Multi-Phase Sequence Labeling." by [Esam Ghaleb](https://esamghaleb.github.io/), [Ilya Burenko](https://www.linkedin.com/in/ilya-burenko-66313825/?originalSubdomain=ru), Marlou Rasenberg, Wim Pouw, Peter Uhrig, Judith Holler, Ivan Toni, [Aslı Özyürek](https://www.mpi.nl/people/ozyurek-asli), [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/). 

## Generate data

Generate data running the following command:
```
python co_speech_gesture_detection/data/generate_data.py
```

This script will generate data for 5-fold cross-validation and store corresponding data to
> co_speech_gesture_detection/data/data/{0,1,2,3,4}/

## Download data and pretrained models

Download pretrained weights for ST_GCN from [Google Drive](https://drive.google.com/file/d/1uC08qBFpYQ7OgXqyVukJsSiEkCsIEnv8/view?usp=sharing), unzip and put them to 
> co_speech_gesture_classification/

Download data from [this folder](https://drive.google.com/file/d/1NvDsff325caHbM_1wGW72PHY8BCDsCXs/view?usp=sharing), unzip and put it to
> co_speech_gesture_classification/data/videos/

After this step the folder should have the following structure:

```
.
├── README.md
├── main_sequential.py
└── co_speech_gesture_detection/
    ├── __init__.py
    ├── sequential_parser.py
    ├── 27_2_finetuned
    ├── config
    ├── data/
    │   ├── data/
    │   │   ├── 0
    │   │   ├── ...
    │   │   └── 4
    │   ├── full_data/
    │   │   └── gestures_info_mmpose.pkl
    │   └── videos/
    │       └── npy3/
    │           └── *.npy
    ├── feeders
    ├── graph
    ├── loss
    ├── model
    ├── processor
    └── utils
```

## Run training

Run training procedure using the follow command:

> python main_sequential.py

## Reference
If you make use of the code or any materials in this repository, please cite the following paper:
```
@inproceedings{ghaleb2023cospeech,
  title={Co-Speech Gesture Detection through Multi-Phase Sequence Labeling},
  author={Ghaleb, Esam and Burenko, Ilya and Rasenberg, Marlou and Pouw, Wim and Uhrig, Peter and Holler, Judith and Toni, Ivan and \"{O}zy\"{u}rek, Aslı and Fernández, Raquel},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={}, % TBC
  year={2024},
  address={WAIKOLOA, HAWAII}, % 
  publisher={IEEE/CVF}, 
  doi={}, % TBC
}

```
