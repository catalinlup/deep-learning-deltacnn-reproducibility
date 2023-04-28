# A step towards reproducing DeltaCNN


Original paper: End-to-End CNN Inference of Sparse Frame Differences in Videos
Paper: https://arxiv.org/abs/2203.03996
Available code by authors: https://dabeschte.github.io/DeltaCNN/

---

Reproducibility project by: 
Crina Mihalache  - 4827333 - F.C.Mihalache@student.tudelft.nl
Catalin Lupau - 5042143 - C.P.Lupau@student.tudelft.nl
Edmundo Sanz-Gadea López - 4553128 - E.Sanz-GadeaLopez@student.tudelft.nl 


Our code: https://github.com/catalinlup/deep-learning-deltacnn-reproducibility

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



<!-- #### Implementation DeltaCNN vs. CNN ''OR'' CNN hyperparameter tuning
 -->

---



### 4. Results

Unfortunately, we were not able to obtain the full set results we set out to achieve due to running into a series of issues, the most important of which were:

* Getting the provided DeltaCNN library to work was inordinately difficult, as it required very specific versions for the GPU driver, CUDA and Pytorch. We spent a lot of time trying to get the library to work. Implementing delta convolutions from scratch was again, not an option, as doing so would have required deep knowledge of gpu-powered parallel programming in CUDA. Such an implementation would have taken a long time.
* Once we were able to get the DeltaCNN library to run on our local machine, we ran into a different issue. It turned out that we didn't have enough GPU memory on our local machine, to run the training process. The machine we used was a Dell XPS 15 9510 with an NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4096 MB of dedicated video memory and 20339 MB of total available graphics memory. Training on the CPU was also not an option, since the provided DeltaCNN library explicitly required the use of a GPU.
* Our next attempt was to try to run our experiments in the cloud. We spent a lot of time trying to obtain a Google Cloud GPU machine, but with little success. Google Cloud required us to make a request to increase our GPU quota. However, once we did that, deploying the cloud machine still did not work, as the resources we would try to allocate were never available. We tried several other cloud providers, until we found a cloud provider called Vultr. We were able to deploy a GPU powered cloud machine on Vultr using our own credit. However, we were not able to install the DeltaCNN library on that cloud machine, due to a mismatch between the required and the installed CUDA version. We did not succeed in changing the driver and CUDA version installed on the Vultr cloud machine due to a lack of system privileges.


As a result of the issues encountered above, we were not able to follow through with our experiments as planned. However, our code for training the networks and running the benchmark is fully functional and, given the necessary resources, it can be run to obtain the full set of results. The code is available on Github and can be accessed through the following link: [DeltaCNN Reproducibility Project](https://github.com/catalinlup/deep-learning-deltacnn-reproducibility).

To compensate for not having experimental results for the DeltaCNN based FasterRCNN, we provide the following graphs showcasing our attempt to optimize the learning rate and weight decay for the version of FastRCNN with MobilenetV2 backbone that uses classical convolutions.

#### 1. Optimizing the learning rate

Training was performed on the video sequence number 4 from the MOT16 dataset. The weight decay (regularization) parameter throughout all of the experiments was set to 5 * 10^-4. Each training session occurred using batches of size 8, 10 epochs and the Adam optimizer. All the frames were resized to a resolution of 96x96 pixels, to save up GPU memory and speed up computations.



![](https://i.imgur.com/MmvNTHP.png)

*Figure 4:* This figure showcases the train loss for a learning rate of 10. We can see that the loss keeps on fluctuating. This suggests that the learning rate is too high and our optimizer overshoots.

![](https://i.imgur.com/eXB67d3.png)

*Figure 5:* This figure showcases the train loss for a learning rate of 10^-7. We can see that the loss is decreasing linearly, rather than exponentially, which suggests that the learning rate is too low and that the optimizer converges too slowly.

![](https://i.imgur.com/oWKEtzU.png)

*Figure 6*: This figure showcases the train loss for a learning rate of 10^-3. It can be seen that the prblems associated with the previous learning rates are avoided and the train loss decreases fast without overshooting.


Looking at the three graphs, we can conclude that a learning rate of 10-7 is too low, a learning rate of 10 is too high and leads to overshooting, while a learning rate of **10^-3** seems reasonable.


#### 2. Optimizing weight decay (regularization)

In this set of experiments, rather than varying the learning rate, we kept the learning rate constant at 10^-3 and we changed the regularization parameter (the weight decay). All of the other parameters were kept the same as previously.

![](https://i.imgur.com/nbdI5S3.png)

*Figure 7*: This figure showcases the train loss for a weight decay of 1. While regularization is supposed to increase the train loss with the promise of better test accuracies, regularization in this case seems to be exaggerated, as the train loss is continuously increasing.

![](https://i.imgur.com/ZIRJtmo.png)

*Figure 8*: This figure showcases the train loss for a weight decay of 0.15.

![](https://i.imgur.com/qbcIG88.png)

*Figure 9*: This figure showcases the train loss when the weight decay is 0 (no regularization).



Figures 7, 8 and 9 show the train loss for different values of the regularization parameter (weight decay). Looking at these 3 figures, we can observe the influence of the regularization parameter on the train loss. If regularization is too high (like in Figure 7), the train loss will overshoot. For an intermediate value of the regularization parameter, like in Figure 8, the train loss converges asymptotically to a value higher than the minimum possible bayesian loss, which is the desired behavior when performing regularization.


#### 3. Evaluation

After evaluating our trained models against the test set - video sequence 02 of the MOT16 dataset - on a cloud machine using an NVIDIA A10 GPU with 24GB of memory, the following results were obtained.



|      Model Type      | Weight Decay | Avg. IOU | Avg. FPS |
|:--------------------:|:------------:|:--------:|:--------:|
| Classic convolutions |       0      |   0.182  |   38.57  |
| Classic convolutions |     5e-4     |   0.160  |   37.46  |
| Classic convolutions |      0.15    |   0.126  |   38.06  |
| Classic convolutions |       1      |   0.123  |   36.02  |
|  Delta convolutions  |       ?      |     ?    |     ?    |

*Table 2*: Showcases the results, i.e. average intersection over unit and average frames per second, obtained as a result of the evaluation.

The results obtained by running the evaluation are presented in the table above. One aspect we can observe is that the weight decay seems to have a negative impact on the IOU. This suggests that the training dataset was representative for the true distribution and, thus, no regularization was needed. No significant difference in the the speed (FPS) can be noticed in the tested model. This makes sense, since all the models that were tested use the same architecture and the same type of convolutions (classical convolutions). Had we been able to train and test the version of the model with delta convolutions, an improvement in speed would have probably been observed.

Another aspect that needs to be discussed is the poor IOU performance of all the models. The poor performance is likely the result of having scaled the images to a very low resolution (96x96). The MOT16 dataset contains many pedestrians that are quite far-away, so they have a small footprint on the image. Drastically reducing the resolution likely made many of the far-away pedestrians unidentifiable or undistinguishable from other pedestrians close to them.


---

### 5. Discussion


Our conclusion is that the chosen paper is **not** easy to reproduce for the following reasons:

- The models proposed by the paper are large and difficult to train without a GPU with enough memory.
- The datasets used for the experiments are very large and difficult to preprocess or load on a remote machine.
- The provided DeltaCNN library is unnecessarily hard to install as it requires very specific versions of CUDA and PyTorch.
- The metrics used as part of the evaluation are not well explained. When reading the paper, it was not clear to us what the PCKh metric was, for example.
- The inner workings of DeltaCNN are explained way too briefly. It required a lot of effort on our side to understand how the proposed technique works.


Having established the main shortcomings, our suggestions to improve reproducibility are the following:

- The creation of a more cross-compatible version of the DeltaCNN library.
- The authors should redo the experimental section of the paper, focusing on a series of small scale experiments that can be run by everyone, while still being able to showcase the advantage of DeltaCNN.
- The paper should provide a concrete example that shows step by step how an input tensor would be processed by a DeltaCNN network.
- The paper should include an appendix that explains all metrics that were used during evaluation, as well as the reasons why these metrics were chosen.
- The experimental section of the paper should include significance testing to show that the obtained results are scientifically sound.

<!-- #### Assesment of paper reproducibility [maybe move before method, to explain why we are implementing totally different things] -->

<!-- List all identified issues and describe individually; Follow "A Step Toward Quantifying Independently Reproducible Machine Learning Research"
 -->

<!-- Issue 1: Setting up the Google Cloud Machine 
Issue 2: CUDA; missing information
Issue 3: Cannot implment Pose-ResNet, only ResNet so far 
Issue 4: Evaluation metric for evaluating perfromance PCKh not explained in the paper, not any additional information on how is it implemented was offered. It is not freely avialble on resources such as TorchMetrics. An implementation has been found on the following public repository however it is not certain the same method is used by the authors [PCKh](https://github.com/ilovepose/fast-human-pose-estimation.pytorch/blob/master/lib/core/evaluate.py) -->




---

### 6. Implementation

The purpose of this section is to provide a brief overview of our implementation (repository) meant to be used during the experimental part of our methodology.

The neural networks used throughout the experiments are defined inside the *neural networks* package and instantiated inside *architectures.py*, which includes a dictionary that contains all of the instantiations. Other files that are important are *train_jobs.py* and *predict_jobs.py*, each containing dictionaries that define the parameters for a training task or a prediction (evaluation task). 

To run a training job, run the following command:

```bash
python train.py <the name of the training job>
```

The model outputted by the training process will be saved in a folder called *saved_models*.

Running a prediction (evaluation) job can be achieved in a similar fashion, running the following command:

```bash
python predict.py <the name of the prediction job>
```


<!-- ### Conclusions
 -->

--- 
### References 
[1. DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos. Mathias Parger, Chengcheng Tang, Christopher D. Twigg, Cem Keskin, Robert Wang, Markus Steinberger, CVPR 2022, June 2022](https://dabeschte.github.io/DeltaCNN/)
[2. Face and hand tracking in the browser with MediaPipe and TensorFlow.js](https://blog.tensorflow.org/2020/03/face-and-hand-tracking-in-browser-with-mediapipe-and-tensorflowjs.html)
[3. Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments. Catalin Ionescu, Dragos Papava, Vlad Olaru and Cristian Sminchisescu. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, No. 7, July 2014](http://vision.imar.ro/human3.6m/pami-h36m.pdf)
[4. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Mingxing Tan, Quoc V. Le. ICML 2019](https://arxiv.org/abs/1905.11946)
[5. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. 2016. ](https://arxiv.org/abs/1506.01497)
[6. Generalized Intersection Over Union: A Metric and a Loss for Bounding Box Regression. Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, Silvio Savarese. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 658-666.](https://openaccess.thecvf.com/content_CVPR_2019/html/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.html)



---
### Task Division

#### Catalin Lupau
- Implementation of the codebase for the experiments.
- Running the compensatory experiments.
- Intepreting the experimental results.
- Working on the blog post.
- Cloud Machine Deployment.


#### Crina Mihalache

- Review of the original paper.
- Doing research on how DeltaCNN works.
- Preparing slides for the weekly TA meetings.
- Note taking and project organization.
- Working on the blog post.


#### Edmundo Sanz-Gadea López
- Installing the project environment.
- Installing all necessary drivers.
- Running code locally, including deltaCNN implementation.
- Working on the blog post.
- Cloud Machine Deployment.