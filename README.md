# Nanodet-YoloV8-Pose-MeterReader

使用Nanodet+YoloV8-Pose实现指针仪表的实时检测、高精度读数识别（借助ncnn框架）

[2024.10.16]更新：
-1.指针仪表检测目标检测和关键点检测数据集：通过网盘分享的文件：指针仪表检测数据集与标注细节.zip
链接: https://pan.baidu.com/s/19I3vs1c01DAdB2V3G67RlA?pwd=383j 提取码: 383j

-2.使用yolov8-pose训练模型时，对部分参数进行了修改，这里给出原始的训练工程：通过网盘分享的文件：ultralytics.zip
链接: https://pan.baidu.com/s/1FG3-ONVURBYpfIqQOGLqrg?pwd=b8jf 提取码: b8jf

## 起

因为接手了一个指针仪表检测项目，而且要求实时性和检测精度都很高，因此便投入指针仪表检测和识别算法的研究。先前我在Github寻找一些灵感，发现有用Yolov5+DeepLabV3Plus来进行读数识别的，我自己也编写了代码实现了一遍，可以通过下面的链接看到我的实现:[YoloX-DeepLabV3Plus-MeterReader](https://github.com/zhahoi/YoloX-DeepLabV3Plus-MeterReader)。

自己亲自实现之后发现，使用语义分割获取表盘和指针的目的是为了获取几个关键的坐标点，分别为指针仪表针尖、指针仪表的表盘中心、指针仪表刻度的起始点和终止点。根据这几个点计算指针指向占据整个量程的百分比，最后乘以仪表盘的量程得到最终的读数。既然如此费劲周折就是为了获取这几个点，为什么不直接获取这几个点，然后直接根据这几个点来计算读数，这样也避免了”中间商赚差价“影响最后的读数精度。

在网上搜索了一番，CSDN上一篇文章给了我一些启发，发现这篇文章使用关键点检测来直接获取我们需要的几个关键坐标点，该文章使用的获取关键点的算法为YoloV8Pose，于是我十分欣喜地想尝试使用该算法来提高读数的精度。

该篇文章：[【计算机视觉】基于YOLOv8的关键点检测的仪表盘读数方案详解](https://blog.csdn.net/nuomuo/article/details/136883680)



## 承

设想先使用轻量化的检测网络来裁剪出包含指针仪表的ROI区域，然后输入到YoloV8-Pose中进行关键点检测，最后根据检测出的关键点来计算最终的读数。在实际标注关键点数据时，发现现有的关键点检测标注是针对单类别来的，而我的设想是分别对指针部分和量程的起始点与终止点分别用目标框标出。又通过一通搜索发现了一个付费专栏里的一篇文章可以进行多类别标注，于是付费了参考了大神的方法，终于解决了标注和训练的问题。付费文章：[【保姆级教程】YOLOv8_Pose多目标+关键点检测：训练自己的数据集](https://blog.csdn.net/m0_51579041/article/details/136820873)。

训练完成后，将训练的Pytorch权重转为ncnn框架下的权重，最后编写了c++代码（本项目仓库），实际测试之后发现检测结果确实比使用目标检测+语义分割的好。但是我也发现YoloV8-Pose针对量程起始点和终止点的检测是准确的，而针对指针仪表中心和指针仪表针尖关键点的准确度却没那么高。

我又在想有没有什么办法可以让指针顶点的坐标准确度更高一点，于是我又发现了一篇使用传统图像处理算法获取仪表指针的方法，配合Yolov8-Pose的结果搭配使用，可以达到最好的效果。[指针仪表检测算法代码](https://blog.csdn.net/qq_39142743/article/details/116164374)



## 转

实际检测流程：

![流程图.png](https://www.pnglog.com/n0Lf6E.png)



实际检测结果：

![微信截图_20240829174428.png](https://www.pnglog.com/Um5Zsn.png)

![1_processed_image.jpg](https://www.pnglog.com/veegqO.jpg)


## 合

该仓库的代码是我最终修改的版本，验证过精度可以达到0.02Mpa左右，检测时间大概在90-140ms之间（没有gpu加速的情况）。如果觉得该代码有用的话，可以给个Star。同时给Star的人可以向我索要训练数据集和数据标注细节。

以上。

## Reference
-[yolov8s-pose-ncnn](https://github.com/Rachel-liuqr/yolov8s-pose-ncnn)
