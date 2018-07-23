

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

DeepLearning Project. 
The base net from the segmentation exercise worked quite well with the default data that was provided.
    1. Got final score of 0.36 pretty with the default dataset that came with the assignment. Close to 0.4 needed for the assignment but not there yet.
    2. Tried the model with the follow me exercise on the Quad copter and it worked fairly well.

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

