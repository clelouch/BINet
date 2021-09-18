# BINet
# [BINet: Bidirectional Interactive Network for Salient Object Detection](https://doi.org/10.1016/j.neucom.2021.09.020)

This repo. is an official implementation of the *BINet* , which has been accepted in the journal *Neurocomputing, 2021*. 

The main pipeline is shown as the following, 
![BINet](figure/BINet.png)

And some visualization results are listed 
![results](figure/results.png)

![modules](figure/visualization.png)

![PR Curves](figure/PR.png)

![F-measure Curves](figure/F-measure.png)

## Dependencies 
```
>= Pytorch 1.0.0
OpenCV-Python
[optional] matlab
```

## Training
pretrained resnet50 is available at ([Baidu](https://pan.baidu.com/s/1K4-b6JPi6E34kgH8gdbYqQ)) [code:nfih]
```
python main.py 
```

## Test
```
 python main.py --mode=test 
```
We provide the trained model file ([Baidu](https://pan.baidu.com/s/17Gfxj0VFkSCTFcFl2mkn1Q)) [code:9p9a]

The saliency maps are also available ([Baidu](https://pan.baidu.com/s/1PIBTcZwre-K8LAu5owLxYA)). [code:ormo]

## Citation
Please cite the `BINet` in your publications if it helps your research:
```
@article{CHEN2021,
  title = {BINet: Bidirectional Interactive Network for Salient Object Detection},
  author = {Tianyou Chen and Xiaoguang Hu and Jin Xiao and Guofeng Zhang and Shaojie Wang},
  journal = {Neurocomputing},
  year = {2021},
}
```
## Reference
[poolnet](https://github.com/backseason/PoolNet)

[BASNet](https://github.com/xuebinqin/BASNet)

## Related works on SOD
[BPFINet](https://github.com/clelouch/BPFINet)
