# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/random_sample.png 
[image2]: ./output/histogram.png 
[image3]: ./downloaded_images/00000_00000.ppm
[image4]: ./downloaded_images/00000_00001.ppm
[image5]: ./downloaded_images/00000_00002.ppm 
[image6]: ./downloaded_images/00000_00003.ppm
[image7]: ./downloaded_images/00000_00004.ppm 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and the project code is in Traffic_Sign_Classifier.html or Traffic_Sign_Classifier.ipynb.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32 32 3]
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I randomly extracted 18 images from the data set. I will use the (normalized) full RGB channels for the dataset.

![image1]

Also, here is a bar chart showing how the data is distributed over all the ground truth labels.

![image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image data to make all RGB channel values between 0 and 1 for fairness among different images. The normalization is done in the `preprocess_imgs` function.

I did not use grayscale images since I think color will provide additional valuable information. For example, stop sign will always be red and the red surface color will boost confidence of the CNN over stop sign.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16       									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		|  Inputs 400, Outputs 120       									|
| RELU					|	
| Dropout		|  Keep probability: 0.5      									|
| Fully connected		|  Outputs 84       									|
| RELU					|	
| Dropout		|  Keep probability: 0.5      									|
| Softmax				|         									|
|																


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I did the following process:

(1) First, I found the baseline LeNet model (batch_size = 128, epoch numbers = 10, learning rate = 0.001) gave me 0.906 validation accuracy, which is a good starting point;

(2) I played with batch_size and found that batch_size = 64 will give me better validation accuracy (0.0931) while further decreasing it will make the model too noisy (i.e., large fluctuation if I plotted validation accuracy against num_epochs). Thus batch_size = 64 is a sweet spot;

(3) There is no regularization in the original LeNet model. In order to avoid overfitting, I introduced two dropout layers right after the two fully connected layers, which gave me improvement of validation accuracy to around 0.95.

(4) Since the batch_size is smaller and it takes longer to train and saturate, I increase the epochs to 15.

(5) Played with learning rate and found the current one is good enough.

Finally, I used batch_size = 64, epoch numbers = 15, learning rate = 0.001, and added 2 dropout layers.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.949
* test set accuracy of 0.932

I chose the LeNet introduced in the course, which is known for its good results on image net dataset on classification, given the similarity of task, After hyperparameter tuning mentioned above, I got my validation accuracy yield monotomic increase. And the test accuracy is 93.2% which is good enough.

 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![image3] ![image4] ![image5] 
![image6] ![image7]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)  		| Speed limit (20km/h)   									| 
| Speed limit (120km/h)    			| Speed limit (120km/h)									|
|No passing for vehicles over 3.5 metric tons					| No passing for vehicles over 3.5 metric tons										|
| Yield      		| Yield				 				|
| Ahead only			| Ahead only     							|

The model was able to correctly guess 5 of the 5 traffic signs. I found that I predicted the results 100% correct. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

One thing a little confusing to me is that for each image, the top 5 softmax probablities are all [1,0,0,0,0], which means that my model is VERY VERY confident about its prediction. 

I first doubted it could be overfitting but I got the same results with or without the dropout layers. Then I doubted that maybe the test images themselves are too good or easy to classify, then I used a random non-sign image, and then I got [1,0,0,0,0] output: the model is too confident even though it's apparently wrong.

Then I doubted that maybe the number of epochs is too large so that the network became overconfident. However, I got something similar like [1, 1e-23, 0, 0, 0] when I decreased num_epoch from 15 to 1. 

Finally, I doubted that maybe it's because I did not transform my images to grayscale and the model became sensitive to the color of the signs, while I don't have time to test or dive into it.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


