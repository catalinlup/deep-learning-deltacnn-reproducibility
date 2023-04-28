# A step towards reproducing DeltaCNN


Original paper: End-to-End CNN Inference of Sparse Frame Differences in Videos   
Paper: https://arxiv.org/abs/2203.03996  
Available code by authors: https://dabeschte.github.io/DeltaCNN/  


---

Reproducibility project by:   
[Catalin Lupau](https://github.com/catalinlup)   
[Edmundo Sanz-Gadea López](https://github.com/sanzgadea)  
Crina Mihalache 

--- 


### Introduction 
Videos have become an essential part of data analysis, and Convolutional Neural Networks (CNNs) have proved essential over recent years to gain an understanding of the actions and the environment by analyzing video frames. However, recording and analyzing large amounts of video information comes at tremendous costs, both in terms of storage to record the data and computational power to process the frames. Therefore, it is interesting to investigate whether the same understanding can be gained without having to process each frame fully. This was the basis for the research paper "End-to-End CNN Inference of Sparse Frame Differences in Videos" by the Graz University of Technology and Meta Reality Labs, published in 2022. The number of applications for such technology is vast, from self-driving cars to human pose estimation, and from improving hardware for human-robot interactions detecting objects to monitoring long-term wheater patterns. Gaining the same understanding of the problem using less data should allow for improved performance in terms of processing power required and computation time.




#### Short introduction on DeltaCNN concept
Video streams typically have neighboring frames that are very similar to each other, differing by only a few pixels as shown in *Figure 1*. To make convolutional neural networks faster, researchers have explored exploiting this property of video streams. One approach is to only propagate the difference between neighboring frames through the network, instead of processing every frame individually. In a recent paper, a novel convolution operation called DeltaCNN was proposed to achieve this. DeltaCNN calculates the difference between adjacent frames and feeds this delta information to the network, resulting in significant computational savings without sacrificing performance. This is useful in this case as one is only interested in the information gained from the movement of the hand not on that arising from perceived movement of the environment. 


![hand](https://3.bp.blogspot.com/-CWTYSEEB3mA/XmfimK9wP1I/AAAAAAAAC0E/wIvHQktx8IEbeB_vbtIEZt3VFNayIFzRACLcBGAsYHQ/s1600/hand_trimmed.gif) 
*Figure 1:* Visualization of keypoint detection and tracking for hand pose [[2](https://blog.tensorflow.org/2020/03/face-and-hand-tracking-in-browser-with-mediapipe-and-tensorflowjs.html)]


<!-- ![](https://i.imgur.com/zlI4ztN.png) -->
![](https://i.imgur.com/VNTlM9z.png)  
*Figure 2:* Working principle of spatially sparse convolutions for videos. Computing the difference between the current and previous input, large parts of convolution input become zero, shown in white. Since zero-valued inputs do not contribute to the output, these values can be skipped to reduce the number of operations.


---

#### Original Paper

The article discusses DeltaCNN, a new framework for accelerating video inference using convolutional neural networks (CNNs). According to the authors, DeltaCNN is the first fully sparse CNN that is optimized for GPUs and applicable to all CNNs without retraining. It reportedly achieves speedups of up to 7x over the dense reference, cuDNN, while maintaining marginal differences in accuracy. [[1](https://dabeschte.github.io/DeltaCNN/)] DeltaCNN is evaluated on two image understanding tasks: human pose estimation and object detection using the Human3.6M, MOT16, and WildTrack datasets. For object detection an EfficientDet architecutre is implemented. For human pose estimation, two different CNN architectures, HRNet and Pose-ResNet, were used, these were initialized with weights obtained from pre-training on ImageNet. Additionally, the authors investigated the speed up performance on three different with different power targets and from different hardware generations: 1) Jetson Nano: a low-end mobile development kit with a power target of 10W and 128 CUDA cores. 2) Dell XPS 9560: a notebook equipped with a Nvidia GTX 1050 with 640 CUDA cores. 3) Desktop PC: a high-end desktop PC equipped with a Nvidia RTX 3090 with 10496 CUDA. In the paper, they only present the performance results for the Human-Pose estimation scenario as shown in *Table 1*.



![](https://i.imgur.com/MGVkpTZ.png)
*Table 1:* Speed and accuracy comparisons of different CNN backends used for pose estimation on the Human3.6M dataset. The same set of auto-tuned thresholds for update truncation is used for all devices and batch sizes b [[1](https://dabeschte.github.io/DeltaCNN/)]. 


---
### 2. Project Goals


The goal of the project was to assess the reproducibility of the paper. To this end, our approach was to build and train two different deep learning models, the first using classical convolutions and the second implementing the delta convolutions introduced by the paper. The two models would then be compared in terms of prediction accuracy and speed in order to assess the validity of the claims the paper makes, i.e. that using the delta convolutions can bring up significant speed up with little or no loss in accuracy.

Our secondary goal, if the first had been realized, would have been to explore the various hyper-parameters of the delta-convolutions and determine their impact on the overall performance, such as speed or accuracy of the model.


<!-- Reproduced: Existing code was evaluated.

Hyperparams check: Evaluating sensitivity to hyperparameters.

##### Initial goal of the reproducibility project: Implement Pose-ResNet to evaluate human psoe estimation on Human3.6M using cuDNN, ours dense and ours sparse on hardware with similar perforamnce as the three devices as avialble from Google Cloud Machines 

### [TO DO add three comaprable machiens] -->


---



### 3. Methodology

To achieve our goals, the following steps needed to be performed:

1. Choosing a dataset to perform experiments on.
2. Choosing a computer vision task to measure the performance on.
3. Choosing a neural network architecture.
4. Implementing that neural network architecture using both classical and delta convolutions.
5. Training the neural networks on the proposed dataset.
6. Performing benchmarks comparing the accuracy and speed of the models.


#### 1) Choosing a dataset to perform experiments on.

One of the first options we considered when it came to choosing the dataset was Human3.6M. Human3.6M is a large dataset of 3.6 million accurate 3D human poses, captured from 5 female and 6 male subjects under 4 different viewpoints, with synchronized image, motion capture, and time of flight(depth) data. It provides a diverse set of human activities and environments for training and evaluating human sensing systems, specifically for human pose estimation models and algorithms.[[3](http://vision.imar.ro/human3.6m/pami-h36m.pdf)] The dataset includes frames that depict the following scenarios: 
Directions, Discussion, Eating, Activities while seated, Greeting, Taking photo, Posing, Making purchases, Smoking, Waiting, Walking, Sitting on chair, Talking on the phone, Walking dog or Walking together. While being a promising option, the large size of the dataset (which would have made training slow, given our hardware limitations) as well as the difficulties encountered while attempting to preprocess it made us steer away from this dataset.


The second option we considered was MOT16 dataset (Multiple Object Tracking Benchmark). This dataset contains a series of video sequences showcasing pedestrians. Due to the fact that this dataset does not require a lot of preprocessing and it is smaller than Human3.6M, we decided to use one video sequence from this dataset as our training data.

#### 2) Choosing a computer vision task to measure performance on.

In the paper, the authors assessed the performance of the delta convolutions on two different computer vision tasks: human pose estimation and object detection. Since our choice in the previous step was the MOT16 dataset, which does not have the poses of the pedestrians labeled, our only option was to go for object detection.

#### 3) Choosing a neural network architecture

In the original paper, the authors used the EfficientDet neural network architecture [[4](https://arxiv.org/abs/1905.11946)] to perform the benchmarks. However, we wanted to go for an option that is easy to implement, yet still relatively fast and easy to train. As a result of this, we chose to use FasterRCNN with a MobileNetV2 backbone. The reason for our choice was that the repository that came with the paper already offered both a  classical and a deltacnn implementation of the MobileNetV2, which we could use. On top of that, rather than training the MobileNetV2 backbone, the feature extractor, from scratch, we could use a pre-trained version of MobileNetV2 and focus on training the rest of the FasterRCNN pipeline.


#### 4) Implementing the neural network architecture using both classical and delta convolutions.


![](https://i.imgur.com/GX9H4Hs.png)
*Figure 3*: Architecture of FasterRCNN [[5](https://arxiv.org/abs/1506.01497)]


FasterRCNN, the architecture that we have chosen to perform object detection consists of several modules, each with its own role:

* **Backbone (feature extractor)** - it takes the raw image as input and output a serties of feature maps used further down the pipeline.
* **Region Proposal Network** - it takes the extracted feature maps as input and outputs the regions in the original image with a high probability of containing an object of interest.
* **RoI pooling** - special pooling mechanism that takes as input the proposed regions as well as the feature maps and ouputs a fixed length feature vector for each region of interest.
* **Classifier** - fully connected neural network that takes as input the feature vector for each of the proposed regions and outputs the class of the object within each region.

As it can be observed from the architecture, the only module of FasterRCNN that involves convolutions is the backbone, specifically the feature extractor. Therefore, changing the classical convolutions to delta convolutions involves swapping the backbone. The two benchmarked architectures we implmented thus shared the same RPN, RoI pooling mechanism and classifier, but different implementations of the feature extractor. We created one version of MobileNetV2 which used classical convolutions, and another version of MobileNetV2 implementing delta convolutions.

#### 5) Training the neural network on the proposed dataset

As is usually the case in object detection, a pre-trained backbone (feature extractor) can be used. Thus, the modules of the network that still require training are the region proposal network and the classifier. 

In our case, since the output of the classical MobileNetV2 is different from the output of the DeltaCNN MobileNetV2, the region propoasal network and the classifier need to be trained separately in each of the two cases.

#### 6) Performing benchmarks comparing the accuracy and speed of the models.

The last step of our methodology is performing a benchmark to compare the speed and accuracy of the two trained models. We propose the use of a different video sequence from the Mot16 dataset for the evaluation.

We propose the following metrics for our benchmark:

* **Average FPS** - the average number of frames per second that the network is able to process.
* **Average IOU** - the average intersection over union score [[6](https://openaccess.thecvf.com/content_CVPR_2019/html/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.html)] for each of the bounding boxes.







<!-- #### Dataset: Human3.6M
Original link : http://vision.imar.ro/human3.6m/description.php
Alternative download: https://blog.csdn.net/qq_42951560/article/details/126380971

Human3.6M is a large dataset of 3.6 million accurate 3D human poses, captured from 5 female and 6 male subjects under 4 different viewpoints, with synchronized image, motion capture, and time of flight(depth) data. It provides a diverse set of human activities and environments for training and evaluating human sensing systems, specifically for human pose estimation models and algorithms.[[3](http://vision.imar.ro/human3.6m/pami-h36m.pdf)] The dataset includes frames that depict the following scenarios: 
Directions, Discussion, Eating, Activities while seated, Greeting, Taking photo, Posing, Making purchases, Smoking, Waiting, Walking, Sitting on chair, Talking on the phone, Walking dog or Walking together. 

Human3.6M was developed by Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu to be used in their research on "Predictive Methods for 3D Human Sensing in Natural Environments", which was published in 2014. The latest published research using this dataset is from 2020. According to the information presented by the authors on their website, all the papers published using this dataset have always had at least one of the original authors involved in the research. The data can be accessed through a request on their website, and it should become available to university students as stipulated. However, although access has been requested for more than a month (at the time of writing this article), the authors have never granted access. Additionally, the authors do not provide any contact point. 


The alternative download dataset has been proven promising at first. By following the instructions, it was possible to download the dataset partially. The total size of the dataset needs to be clearly stated on the website of the authors and on any of their papers; however, it is believed to be at least 20GB. The alternative download failed, and the two main issues will be addressed separately
1. Insufficient resources on the local computer to download the complete dataset. An alternative solution was to set up a cloud machine and directly upload the dataset. However, several issues associated with setting up the cloud machine prevented this solution, as will be explained later on 
2. Two issues have been identified for the partial dataset downloaded on the local resources: some of the labels were missing, and some other figures were mislabelled. 

Provided this was not the original source for the Human3.6M dataset, it is not surprising that such issues would occur. However, it made the used of this dataset during the project impossible. -->


### 4. Results

Unfortunately, we were not able to obtain the full set results we set out to achieve due to running into a series of issues, the most important of which were:

* Getting the provided DeltaCNN library to work was inordinately difficult, as it required very specific versions for the GPU driver, CUDA and Pytorch. We spent a lot of time trying to get the library to work. Implementing delta convolutions from scratch was again, not an option, as doing so would have required deep knowledge of gpu-powered parallel programming in CUDA. Such an implementation would have taken a long time.
* Once we were able to get the DeltaCNN library to run on our local machine, we ran into a different issue. It turned out that we didn't have enough GPU memory on our local machine, to run the training process. The machine we used was a Dell XPS 15 9510 with an NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4096 MB of dedicated video memory and 20339 MB of total available graphics memory. Training on the CPU was also not an option, since the provided DeltaCNN library explicitly required the use of a GPU.
* Our next attempt was to try to run our experiments in the cloud. We spent a lot of time trying to obtain a Google Cloud GPU machine, but with little success. Google Cloud required us to make a request to increase our GPU quota. However, once we did that, deploying the cloud machine still did not work, as the resources we would try to allocate were never available. We tried several other cloud providers, until we found a cloud provider called Vultr. We were able to deploy a GPU powered cloud machine on Vultr using our own credit. However, we were not able to install the DeltaCNN library on that cloud machine, due to a mismatch between the required and the installed CUDA version. We did not succeed in changing the driver and CUDA version installed on the Vultr cloud machine due to a lack of system privileges.


As a result of the issues encountered above, we were not able to follow through with our experiments as planned. However, our code for training the networks and running the benchmark is fully functional and, given the necessary resources, it can be run to obtain the full set of results.

### References 
[1. DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos. Mathias Parger, Chengcheng Tang, Christopher D. Twigg, Cem Keskin, Robert Wang, Markus Steinberger, CVPR 2022, June 2022](https://dabeschte.github.io/DeltaCNN/)  
[2. Face and hand tracking in the browser with MediaPipe and TensorFlow.js](https://blog.tensorflow.org/2020/03/face-and-hand-tracking-in-browser-with-mediapipe-and-tensorflowjs.html)  
[3. Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments. Catalin Ionescu, Dragos Papava, Vlad Olaru and Cristian Sminchisescu. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, No. 7, July 2014](http://vision.imar.ro/human3.6m/pami-h36m.pdf)  
[4. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Mingxing Tan, Quoc V. Le. ICML 2019](https://arxiv.org/abs/1905.11946)  
[5. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. 2016. ](https://arxiv.org/abs/1506.01497)  
[6. Generalized Intersection Over Union: A Metric and a Loss for Bounding Box Regression. Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, Silvio Savarese. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 658-666.](https://openaccess.thecvf.com/content_CVPR_2019/html/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.html)
