

### Discuss Network Architecture.
---------------------

### Convey the understanding of the network architecture

#### Explain purpose of each layer

**The convolution layers**. Here we use depth separable convolution layers. Depth separable have much less parameters than a regular convolution. Convolution are the main powerhorse of neural nets. Convolution when applied to input can bring out the interesting features which can then be used by subsequent layers to compute even more complex relevant features for the task at hand. Each convolution has a weight matrix associated with it, which is what we end up learning when we do the backprop through the loss.

**Depth Separable Filters**
Depth separable filters use convolution at each channel separately. Applying regular convolution with 3x3 kernels for a RGB image with 32 output channels will have 3x3x3*32 = 288 parameters. A 3X3 depth separable filter will apply 3X3 to R,G and B each, giving 3x3 + 3x3 + 3x3 = 27 parameters. The output of this will be a 3 channel layer. The separable convolution then applies 1x1 convolution across channels giving 3x32 = 96 parameter for a total of 27+96 = 123 parameters. Less than half the parameter of regular convolution. Hence these nets can be very useful. 

**Batchnorm layers** Like we normalize the input, batchnorm serves to normalize the input at every intermediate layer. This prevents the net from covariate shift and helps with backpropagation and training. Also, even though batchnormalization makes training slower, it allows the network to be much more stable to initialization and also one can use higher learning rate. It also improves generalization.

**Max Pooling layer** Max Pooling is used to take the maximum in a certain spatial region (2X2). The main idea is that if certain feature is a local neighborhood has a high value, we can use that for the next layers and so. This has two affects. First it build robustness to small translation at each layer, which may be important for classification network. Secondly, it reduces the spatial dimension of the layer's output. This forces the network to learn meaningful concepts at higher level -- with reduced spatial dimension we can still capture the concept. However, it has a downside of loosing spatial information, which can hurt in applications where precise spatial location is important. 

**1x1 Convolution**
1x1 convolution act as matrix multiplication between the input and output layer at each pixel. The weights are the same for all pixels across spatial region. Normally earlier networks use to have the penultimate layer of classification/encoder network flattened and fully-connected to output layer. This has certain downsides for example the network will only work on certain image size. Also, fully connected layer tend to have large number of parameters making it harder to train. 1x1 convolution make the network work with any size images and also has much fewer parameters and achieves good accuracy. 1x1 convolutions make the network fully convolutional and are quite useful for pixel level segmentation networks. See below for more details. 

**Skip Connections**
Skip connections provide information to the decoder from the earlier layer of the network. This results in better reconstruction of the segmentation mask. This can be helpful to create sharp boundaries and also segment out smaller objects.

### Network Design

Tested with couple of different network design. All the design included convolution followed by pooling. I call this as one layer. We are using depth separable convolution and skip connections as suggested in the segmentation lab. They provide many benefits which are discussed above. In this section we will see how added more depth effects the performance of the network 

For the discussion below, final_channels refer to the size of 1x1 convolution and filter refers to the channels of each of the convolution layer. The decoder network has same size channels but in reverse. Stides is fixed at 2.

**2 Layer Net**
 2 Layer net with filters=[32, 64], final_channels = [64]. The networked surprising well with final_score of 0.37. Please see figures below. It struggled to get larger object both target and non-targets and this makes sense since the field of view of 2 layer net isn't as large. 

1. Loss: <img src="images/2_layer_net/loss.png" width="400" >
2. Metric: <img src="images/2_layer_net/performance_2_layer_net.png" width="400" >

3. Samples: 
<img src="images/2_layer_net/following.png" width="200">  <img src="images/2_layer_net/target.png" width="200">  <img src="images/2_layer_net/non-target.png" width="200" >


**3 Layer Net (Best Results)**
To improve the performance of the net I added extra layer. This way the net can see larger fov which will help with accurately segmenting larger objects in the scene. The network performs pretty well with final score of 0.429
filters=[32, 64, 128]
final_channels = [128]
Please see the sample of results below. 

1. Loss: <img src="images/final_results/loss.png" width="400" >
2. Metric: <img src="images/final_results/performance_3_layer.png" width="400" >

3. Samples: 
<img src="images/final_results/following.png" width="200">  <img src="images/final_results/small_target.png" width="200">  <img src="images/final_results/non-target.png" width="200" >


**4 Layer Net** 
To see if adding further layers will help the performance, I used 4 layer net. The net has even larger fov and it works pretty well. However, the performance was slightly worse that the 3 layer net with final score of 0.423
filters=[32, 64, 128, 256]
final_channels = [256]
strides=[2]

1. Loss: <img src="images/4_layer_net/loss.png" width="400" >
2. Metric: <img src="images/4_layer_net/performance_4_layer.png" width="400" >

3. Samples: 
<img src="images/4_layer_net/following.png" width="200">  <img src="images/4_layer_net/target.png" width="200">  <img src="images/4_layer_net/non_target.png" width="200" >

We choose the 3 layer net as the final network choice. 

### Hyper-Parameters

The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

**Epoch:** Tried with few epochs at first. From the graph it seems that epoch of 2 and 3 the validation loss is same as training loss. However, the network performance wasn't all that good. On the other extreme I tried running with 50 epochs. You can see that after around 20 to 30 epochs the validation loss is not decreasing and infact is getting higher (not by a lot but still) that means the network is overfitting. Based on this I have found a epoch of 25 may be a good stopping point. 
![epochs](images/epoch.png)

**Learning Rate:** I played with different learning rates. The learning rate of 0.01 worked out well in practice. Since the validation and training loss was still jumping, I tried to lower the learning rates to 0.001 and 0.0001. At the lower end 0.0001 made the training much slower though it produced smoother graphs. Also the validation error at the end was higher. 

LR=0.0001 <img src="images/low_learning_rate/low_learning_larger_network_0.0001_loss.png" width="300" >   LR=0.001 <img src="images/low_learning_rate/low_learning_0.001_loss.png" width="300">


LR=0.01<img src="images/epoch.png" width="300">   LR=0.1<img src="images/low_learning_rate/learning_rate_0.1_bz_16.png" width="300">
 
 I chose the learning rate of 0.01 worked best. It has good validation accuracy and converged faster. 
                   

**Batch Size:** Normally a larger batch size is better and is constraint by the memory your gpu has. Also, there is a sweet spot in terms of computation speed/efficiency. Low batch size of 1 is generally not advisable. I tried many different size with learning rate fixed to 0.01. Varying the size of images didn't effect the speed of running through epoch, however the batch size of 16 seemed to work the best. Network was converging slowly with higher batch size. Learning rate depends on the batch size. Please see below for more info. For this experiment I used only 1600 of training images. 
Batch size 8:  <img src="images/batch_size/batch_size_8.png" width="300">  Batch size 16: <img src="images/batch_size/batch_size_16.png" width="300"> 

Batch size 32: <img src="images/batch_size/batch_size_32.png" width="300">  Batch size 64: <img src="images/batch_size/batch_size_64.png" width="300"> 

**Learning rate and batch size**
Learning rate and batch size are somewhat dependent on each other. Learning rate of 0.01 may be high for smaller batch size than for larger batch size and may be small for a batch size of 32 or 64. Doing detailed analysis of both will be hard, but please see this interesting article on this https://miguel-data-sc.github.io/2017-11-05-first/. 

As an example, if we reduce the learning rate to 0.01 for batch size of 8 the results improve see figure. 

Learning rate and Batch size ![learning rate batch size 8](images/low_learning_rate/learning_and_batch_size_8.png)


### 1x1 Convolutions

Unlike having fully connected layer, 1x1 convolution provide many benefits compared. First it allow us to run the network on any size images. Also, unlike fully connected network it has much fewer parameters and achieves good accuracy. When netweork uses fully connected layer, they flatten the last layer and then connect with outputs. Because of this flattening the network  ends up hardcoding the relative pixel location in the penultimate layer (that has been flattened). That is top-left pixel has entirely different weights than the middle pixel and the bottom-right pixel. This isn't necessarily the cases for images, which are somewhat translation invariant. 

1x1 convolution uses the same weight irrespective of the location of the last layer of encoder/penultimate -- providing translation invariance. 1x1 convolution learns the dependencies between the input and output channels irrespective of the pixel location. They can be often used to compress the channel size -- reduce redundancy and to develop more complex function using the individual channel values.

#### Reasons for encoding / decoding images

1. Encoder network is a stack of convolution and pooling layers that are added on top of each other. The main idea of the encoder network is to learn the increasing complex set of features that can be used to understand the scene or classes that we are trying to segment. Encoder network layers reduces in spatial size and increase in terms of channels. They sacrifice the spatial information for more complex features that can detect background from hero and other people. 

2. Decoder networks on the other hand that these abstract features which are low in spatial dimension and create segmentation mask of the same size as the original image. One can think of the decoder network as an upsampling layer, which uses the coarse level segmentation at the top most layer of the encoder and create a high resolution segmentation mask. The decoder network can have connection from earlier layers using skip connection or summation operation to get fine level of segmentation. 
    
### Model and data working well for other objects (dog, cat, car, etc.) 

The model is trained on detect target object from non-target (other people) and the background. Since the model has been trained on detecting people and target object, this particular model won't work well objects it has been trained. So it won't work well on detecting cat, dog or car. However, if the model is trained with this data, it can learn to detect these objects. We may have to increase the depth of the network since the size of the object may vary from small cars to very larger car near the camera. Without training on these classes, it is hard to say what the network will do and perform on cat, dog and cars. 

### Challenges 

I started with base net from the segmentation exercise with  the following architecture
   - encoder_layers =[16, 32, 64, 96, 128]
   - 1x1 convolution = [128] # single layer
   - decoder is just reverse of encoder

This worked quite well. Got a final score of 0.36, however I wasn't able to improve its performance to the level needed for this project. To remedy it, I had to take more methodical approach, after trying bunch of different architectures in hacky and unorganized way.

#### Actions to Remedy the Issues
It was getting harder to keep tabs and figure out improvement in an organized way. So I did a couple of things to fix it.  

1. So added code to dump the network architecture, hyper-parameters and the scores of the experiments I was running. 

2. Collected data, but only after analysis of where the model is failing. Earlier attempts to just add data didn't help. Later went back to default dataset and saw that my model wasn't working as well on smaller targets. Added a sample of around 1200 images where the target was small with other people in the scene. 

3. Anaylzed the scores and realized that the peformance for small target is still not as good. Reduced the size of the network, so the earlier layers can have more say. The small targets were getting lost in the later layers The size of the dataset and the fact that smaller targets were not getting detected quite bad and I need perhaps more data on this side. This has been driving the performance low.

4. Realized that I need to be methodical.
   - Write down the params and the performs numbers to see how well I am doing with changing the architecture
   - Create a better dataset. With lots of small target 

# Final Model and Results

**Architecture of the Final Model**
Here is the final architecture of the model that performed the best. It has 3 layer in encoder and 3 layers in decoder. filter here referes to the channels of the layers. final_channels refer to the size of 1x1 convolution The decoder network has same size channels but in reverse. Stides is fixed at 2.

filters=[32, 64, 128]
final_channels = [128]
* Architecture illustration ![arch](images/final_results/network_architecture.png)


**Results on following target:**
![following](images/final_results/following.png)

**Results on non-target:**
![non-target](images/final_results/non-target.png)

**Results on small-target with other people:** 
![target](images/final_results/small_target.png)

**Training and Validation Loss:** 
![training_validation_loss](images/epoch.png)

**Performance number:** 
![Performance](images/final_results/performance_numbers.png)

**Final Model Weights**
Model weights are in the folder final_model in the project directory. 
* hdf5 file: [model_weights](model_weights_run_gpu_12.h5)
* model config file: [model_weights](config_model_weights_run_gpu_12)

**Final HTML file**
Please see the .html file for more details. 
[model_training.html](/outputs/final_final_model_training.html)


### Future Work

The following steps can further improve the model's performance. 

1. The data we have is still quite small. Increasing the dataset will improve the performance of the network. A smart way to add data will be to analyze where the model is struggling and add more data to address those failure/below-par performance cases. This can be iteratively done. Also, there may be data imbalance -- for example having fewer large people vs small targets. 

2. I tried deeper network and even though the numbers were as good playing more with hyperparameters and reducing the size of the layers and training on larger data and more epochs can improve the model.

3. Even though the model is tested in simulating model, adding data augmentation such as perspective/affine distortions as well as some noise and brightness changes can possibly improve the performance of the model. 

4. Including other classes other than people can also possibly improve the networks performance on detecting target. 

5. We can also use a pretrained network which have been trained on large dataset and finetune on this smaller dataset. Using these pretrained model can further improve the performance. 
