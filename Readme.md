# RiR-Tensorflow
[![Packagist](https://img.shields.io/badge/Tensorflow-1.3.0-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Tensorlayer-1.6.1-blue.svg)]()

Abstract
---
This is the simplified implementation of the paper - resnet in resnet: generalizing residual architectures[1]. This project adopts 3 different models: usual CNN, ResNet and ResNet in ResNet (RiR) structure. Moreover, I use two dataset to train the models. After surveying, no other tensorflow implementation can be found. As the result, I write this program to evaluate if the idea of RiR is practical. There's another chainer implementation [here](https://github.com/nutszebra/resnet_in_resnet). Besides, I provide the paper [link](https://arxiv.org/abs/1603.08029) as well. 

Structure
---

* Left: the usual CNN
* Middle: ResNet
* Right: RiR    

![](https://github.com/SunnerLi/rir/blob/master/img/structure.jpg)

<br/>

Result
---
First, I use CIFAR10 just like the original paper descript. I train for 4 epoches, and 400 random bagged image are selected in each epoches. The result is shown below:    

![](https://github.com/SunnerLi/rir/blob/master/img/cifar_400_4.png)

However, I also use MNIST to train the models. 2 epoches are adopted and 200 random bagged image are selected in each epoches. The result is shown below:

![](https://github.com/SunnerLi/rir/blob/master/img/MNIST_400_2.png)


Conclusion
---
As you can see in the first result image, the RiR structure really does the good job. The ResNet in ResNet structure not only learns the identity mapping but also learns the residual concepts. However, it shows the worse performance in MNIST. The reason I guess is that the RiR structure cannot do very well in the images which don't have complex feature formation. As the result, the usual CNN might be a great model in some cases. The more argument can be found in the ultra deep structure.     
For conclusion, the concept of RiR is still a creative idea to combine the property of two traditional methods.     

Reference
---
[1]	S. Targ, D. Almeida, and K. Lyman, “RESNET IN RESNET: GENERALIZING RESIDUAL ARCHITECTURES,” _arxiv,_ 2016.

