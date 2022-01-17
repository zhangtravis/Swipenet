# SWIPENet
A novel neural network architecture for dealing with small object detection and noise data in the underwater scenes. Related paper has been accepted by the conference International Joint Conference on Neural Networks (IJCNN) 2020.

Chen, Long, Liu, Zhihua, Tong, Lei, Jiang, Zheheng, Wang, Shengke, Dong, Junyu, and Zhou, Huiyu. "Underwater object detection using Invert Multi-Class Adaboost with deep learning", Proc. of International Joint Conference on Neural Networks (IJCNN), Glasgow, UK, 19-24 July, 2020.

# Abstract
In recent years, deep learning based methods have achieved promising performance in standard object detection. However, these methods lack sufficient capabilities to handle underwater object detection due to these challenges: (1) Objects in real applications are usually small and their images are blurry, and (2) images in the underwater datasets and real applications accompany heterogeneous noise. To address these two problems, we first propose a novel neural network architecture, namely Sample-WeIghted hyPEr Network (SWIPENet), for small object detection. SWIPENet consists of high resolution and semantic rich Hyper Feature Maps which can significantly improve small object detection accuracy. In addition, we propose a novel sampleweighted loss function which can model sample weights for SWIPENet, which uses a novel sample re-weighting algorithm, namely Invert Multi-Class Adaboost (IMA), to reduce the influence of noise on the proposed SWIPENet. Experiments on two underwater robot picking contest datasets URPC2017 and URPC2018 show that the proposed SWIPENet+IMA framework achieves better performance in detection accuracy against several state-of-the-art object detection approaches. 
# Qualitative comparison of SWEIPENet and SSD
![](https://github.com/LongChenCV/SWIPENet/blob/master/Comparison.png) 
# Dependencies
* python >= 3.5
* keras
* numpy >= 1.18.0
* scipy
# DataSets
The underwater robot picking contest datasets is organized by National Natural Science Foundation of China and Dalian
Municipal People’s Government. The Chinese website is http://www.cnurpc.org/index.html and the English website is http:
//en.cnurpc.org/. The contest holds annually from 2017, consisting of online and offline object detection contests. In this
paper, we use URPC2017 and URPC2018 datasets from the online object detection contest. To use the datasets, participants need to communicate with zhuming@dlut.edu.cn and sign a commitment letter for data usage: http://www.cnurpc.org/a/js/2018/0914/102.html
# Usage
**Training stage:**
1. Train the first detection model:
```python ssd512_training.py```
2. Update the weights using IMA algorithm:
```python ssd512_updateweight.py```
3. Train the second, third ... detection models using 1 and 2.

**Testing stage:**  
```python ssd512_evaluation.py```

# Citation
If you use these models in your research, please cite:
```
@ article{LongChenCV,  
	author = {Chen Long, Liu Zhihua, Tong Lei, Jiang Zheheng, Wang Shengke, Dong Junyu and Zhou Huiyu},  
	title = {Underwater object detection using Invert Multi-Class Adaboost with deep learning},  
	journal = {Proc. of International Joint Conference on Neural Networks (IJCNN)},  
	year = {2020}  
} 
```

# Swipenet
