[TensorFlow 2] ArcFace: Additive Angular Margin Loss for Deep Face Recognition
=====

## Related Repositories
<a href="https://github.com/YeongHyeon/ResNet-TF2">ResNet-TF2</a>  

## Concept
<div align="center">
  <img src="./figures/arcface.png" width="700">  
  <p>Concept and Pseudo-code of ArcFace [1].</p>
</div>

## Performance

|Indicator|Value|
|:---|:---:|
|Accuracy|0.98100|
|Precision|0.98101|
|Recall|0.98083|
|F1-Score|0.98090|

```
Confusion Matrix
[[ 961    1    2    0    2    2    5    4    2    1]
 [   2 1118    4    1    0    0    4    6    0    0]
 [   1    1 1020    2    0    0    0    7    1    0]
 [   1    0    3  995    0    4    2    1    2    2]
 [   0    0    1    0  965    0    2    2    3    9]
 [   1    1    0   10    1  871    4    2    1    1]
 [   7    2    3    2    6    1  936    0    1    0]
 [   1    2    9    1    0    0    1 1012    1    1]
 [   1    0    8    0    2    4    1    3  949    6]
 [   1    1    1    2    7    4    1    8    1  983]]
Class-0 | Precision: 0.98463, Recall: 0.98061, F1-Score: 0.98262
Class-1 | Precision: 0.99290, Recall: 0.98502, F1-Score: 0.98894
Class-2 | Precision: 0.97050, Recall: 0.98837, F1-Score: 0.97936
Class-3 | Precision: 0.98223, Recall: 0.98515, F1-Score: 0.98369
Class-4 | Precision: 0.98169, Recall: 0.98269, F1-Score: 0.98219
Class-5 | Precision: 0.98307, Recall: 0.97646, F1-Score: 0.97975
Class-6 | Precision: 0.97908, Recall: 0.97704, F1-Score: 0.97806
Class-7 | Precision: 0.96842, Recall: 0.98444, F1-Score: 0.97636
Class-8 | Precision: 0.98751, Recall: 0.97433, F1-Score: 0.98088
Class-9 | Precision: 0.98006, Recall: 0.97423, F1-Score: 0.97714

Total | Accuracy: 0.98100, Precision: 0.98101, Recall: 0.98083, F1-Score: 0.98090
```

## Requirements
* Python 3.7.6  
* Tensorflow 2.1.0  
* Numpy 1.18.1  
* Matplotlib 3.1.3  

## Reference
[1] Jiankang Deng et al. (2018). <a href="https://arxiv.org/abs/1801.07698">ArcFace: Additive Angular Margin Loss for Deep Face Recognition</a>. arXiv preprint arXiv:1801.07698.
