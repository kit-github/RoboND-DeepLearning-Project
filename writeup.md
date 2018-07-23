

Network Architecture.
---------------------
Segmentation Lab.

0. The segmentation network has the following layers
   - input is (128, 128, 3)
   - batch_size 8
   - learning rate 0.001
   - encoder_layers =[16, 32, 64, 96, 128]
   - 1x1 convolution = [128] # single layer
   - decoder is reverse of [96, 64, 32, 16, 3]


Convey the understanding of the network architecture
-----------------------------------------------------
Explain purpose of each layer
1. The convolution layers. Here we use depth separable convolution layers. Depth separable have much less parameters than a regular convolution. Convolution are the main powerhorse of neural nets. Convolution when applied to input can bring out the interesting features which can then be used by subsequent layers to compute even more complex relevant features for the task at hand. Each convolution has a weight matrix associated with it, which is what we end up learning when we do the backprop through the loss.

Depth separable filters use convolution at each channel separately. if we were to apply 3x3 kernel for a RGB image followed by a 32 channels. Aregular convolution will have 3x3x3*32 = 288 parameters. 

A 3X3 depth separable filter will apply 3X3 to R, and 3x3 applied to G and 3x3 applied to B. Hence it has 3x3 + 3x3 + 3x3 = 27 parameters. The output of this will be a 3 channel layer. The separable convolution then applies 1x1 convolution acrros channels giving 3x32 = 96 parameter for a total of 27+96 = 123 parameters. Less than half the parameter of regular convolution. Hence these nets can be very useful. 


2. Batchnorm layers - Like we normalize the input, batchnorm serves to normalize the input at every intermediate layer. This prevents the net from covariate shift and helps with backpropagation and training. Also, even though batchnormalization makes training slower, it allows the network to be much more stable to initialization and also one can use higher learning rate. It also improves generalization.

3. Max Pooling layer - is used to take the maximum in a certain region (2X2). As a result it reduces the spatial size. 

4. 1x1 convolution.
They are used to make the network fully convolutional. Normally one would take the last layer and connect with a fully connected layer. 

5. Skip connections
Skip connections provide information to the decoder from the earlier layer of the network. This results in better reconstruction of the segmentation mask. This can be helpful to create sharp boundaries and also segment out smaller objects.

DeepLearning Project. 
The base net from the segmentation exercise worked quite well with the default data that was provided.
    1. Got final score of 0.36 pretty with the default dataset that came with the assignment. Close to 0.4 needed for the assignment but not there yet.
    2. Tried the model with the follow me exercise on the Quad copter and it worked fairly well.

Hyper-Parameters
---------------
The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

Epoch - Tried with few epochs. The performance wasn't good. 
Learning Rate - Low learning rates took much longer. 
Batch Size - batch size of 8 and 16. Didn't try higher

1x1 convolutions and when are they needed. 
-----------------
They are used to make the network fully convolutional. Normally one would take the last layer, flatten it and then connect with a fully connected layer. However, in flattening it and connecting it we hardcode the pixel location -- top-left pixel is now the first neuron and bottom-right the last most. The network loose the translation invariance. Also the kernel in the fully-connected layer can only work for certain size images since the size of the flattened layer can't change.

On the other hand 1x1 convolution uses the same weight irrespective of the spatial location of the last layer of encoder. Hence, if the image is larger than the last larger of encoder will be larger but since 1x1 convolution works on depth, it isn't effected by it. The output of 1x1 will be larger size also. 1x1 convolution learn the dependencies between the channels irrespective of the pixel location. Making them translation invariant. They can be used to compress the channel size -- reduce redundancy and to develop more complex function using the individual channel values.

Reasons for encoding / decoding images
---------------------------------------
1. Encoder is learning features that can be used to detect objects. Normally we will have a class at the end.
2. Decoder is used to recreate the segmentation mask from these features. The skip connections 

Attempts to improve network architecture
-----------------------------------------
    3. Tried to improve the network but didn't work as well.
       - running on more epochs.
       - increasing the layer sizes. 

    4. Added more data and tried. The results still didn't improve.

    5. Reduced the training rate. But that didn't help.

    6. The model seems to be overfitting. How to add regularization loss in keras.

    7. Early stopping -- to prevent overfitting

Issues:
------


1. Tried bunch of architecture with larger hidden layers and learning rate but seemed it was making the performane worse.
nothing was working.

2. Collected data. Hoping more data will help. Took suggestions. Added 1500 images to the existing training dataset. It didn't help as much either

3. Anaylzed the scores and realized that the peformance for small target is quite bad and I need perhaps more data on this side. This has been driving the performance low.

4. Realized that I need to be methodical.
   - Write down the params and the performs numbers to see how well I am doing with changing the architecture
   - Create a better dataset. With lots of small target 

