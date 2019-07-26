# Referring Image Segmentation via Recurrent Refinement Networks

This code implements the model described in [Referring Image Segmentation via Recurrent Refinement
Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Referring_Image_Segmentation_CVPR_2018_paper.html),
CVPR 2018. 

## Setup

This code derives from [TF-phrasecut-public](https://github.com/chenxi116/TF-phrasecut-public).
Please follow its `Setup` and `Data Preparation` sections except that the DeepLab-ResNet backbone
comes from [tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet). The
pre-trained DeepLab-ResNet model can be downloaded from
[here](https://github.com/DrSleep/tensorflow-deeplab-resnet#caffe-to-tensorflow-conversion).

## Usage
Before training, make sure `refer`, `cocoapi`, and `deeplab` are in `PYTHONPATH`.
```
export
PYTHONPATH=./external/refer:./external/cocoapi/PythonAPI:./external/tensorflow-deeplab-resnet:$PYTHONPATH
```

To train a model on UNC dataset, run
```
python main_convlstm_p543.py -m train -d unc -t train -f ckpts/unc
```

To test the model with Dense CRF, run
```
python main_convlstm_p543.py -m test -d unc -t testA -f ckpts/unc -i 700000 -c
```

## Cite
If you use this code, please consider citing
```
@inproceedings{li2018referring,
  title={Referring Image Segmentation via Recurrent Refinement Networks},
  author={Li, Ruiyu and Li, Kaican and Kuo, Yi-Chun and Shu, Michelle and Qi, Xiaojuan and Shen,
      Xiaoyong and Jia, Jiaya},
  booktitle={CVPR},
  year={2018}
}
```
