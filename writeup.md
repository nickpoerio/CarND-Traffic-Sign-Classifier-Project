# **Traffic Sign Recognition** 

## Writeup

[//]: # (Image References)

[image1]: ./writeup_resources/exploratory_visualization1.png "training set"
[image2]: ./writeup_resources/exploratory_visualization2.png "validation set"
[image3]: ./writeup_resources/exploratory_visualization3.png "test set"
[image4]: ./writeup_resources/probability_distributions.png "classes occurrences"
[image5]: ./writeup_resources/augmentation.png "augmentation"
[image6]: ./writeup_resources/internet_images.png "internet_images"
[image7]: ./writeup_resources/internet_softmax.png "internet_softmax"
[image8]: ./writeup_resources/activation.png "activation"


---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 by 32 by 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Each row includes 6 randomly chosen images.

![alt text][image1]
![alt text][image2]
![alt text][image3]

There is also a histogram chart, summarizing the distribution of the classes occurences in the different sets, which appears to be very similar.

![alt text][image4]

Hence, the split is ok.
Anyhow, some classes occur much less in absolute terms, so I augmented more such classes in order to have a uniform training set(see next).


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First I applied grayscale to images: color information doesn't seem to be extremely meaningful in order to discern from a sign to another
Then I applied image effects in order to augment the training set.
In particular, I applied the following transformations, randomly:

1. translation
2. rotation
3. perspective transformation
4. zoom

From a quick look at the samples there seemed to be already enough blurring and brightening/darkening effects. 
I augmented at least once  each training sample. Less numerous classes have been augmented in order to have a similar number of samples.

![alt text][image5]

As a last step, I normalized the image data in order to have a better numerical conditioning of the problem.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the LeNet architecture with dropout, defined as following:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				| keep probability 0.8							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout				| keep probability 0.8							|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	 				|
| Flatten				| outputs 400 									|
| Fully connected		| outputs 120 									|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| Fully connected		| outputs 84 									|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| Fully connected		| outputs 43 									|
| Softmax				| 	        									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:

1. optimizer: ADAM
2. batch size: 128
3. epochs: 50
4. learning rate: 0.001
5. keep probability for convolutions: 0.8
6. keep probability for fully connected layers: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.968
* validation set accuracy of 0.953 
* test set accuracy of 0.941

I decided to keep the canonical LeNet architecture, as since the first tests it didn't show significant underfitting (good accuracy on the training set). Besides, the MNIST problem is not too much different from this one.  return
On the other hand, it was immidiately evident the need for regularization (poor validation accuracy).  return
I first tryed to add dropout just to the fully connected layers (as in the original dropout paper), but, despite a significant improvement, some overfitting was evident (still significant difference between training and validation accuracy).  return
Surfing the net for papers, I found that a slight dropout is recommended also for the convolutional layers, in order to avoid model noise propagation to the deeper levels.  return
The final model shows almost absence of overfitting. Some improvement could now be performed in accuracy, using bigger/deeper networks.  return
The detailed analysis of the precision/recall of the single classes helps to find specific issues related e.g. to the training set (to be further and thoroughly augmented for those classes).

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] 

The third image has been chosen on purpose to be extremely "zoomed out" in order to see how much the model is able to generalize in such condition. During augmentation, I applied some zoom effect, but not so extreme.
The other images shouldn't be tricky.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30 km/h)	| Speed limit (30 km/h)							| 
| Stop	     			| Stop	 										|
| Speed limit (70 km/h)	| Yield											|
| Turn right ahead 		| Turn right ahead      		 				|
| Road work				| Road work		      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. As forecasted, the third image is quite tricky.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here's the first 5 softmax probabilities for each prediction (rounded at the 4th decimal): apart from figure #3, the other predictions are extremely accurate.
The model is still a little bit "shortsighted" :). It should be trained with more distant images. In fact, currently, the precision and recall on the "Speed limit (70 km/h)" class appear to be very good even on the test set!

Image #1: Speed limit (30 km/h)	

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9985        			| Speed limit (30 km/h)							| 
| .0013    				| Speed limit (50 km/h)							|
| .0002					| Speed limit (70 km/h)							|
| .0000	      			| Speed limit (20 km/h)			 				|
| .0000				    | Speed limit (80 km/h)	     					|

Image #2: Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000       			| Stop											| 
| .0000    				| No Entry										|
| .0000					| Keep right									|
| .0000	      			| Priority road					 				|
| .0000				    | Turn left ahead		     					|

Image #3: Speed limit (70 km/h)	

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .1968       			| Yield											| 
| .1870    				| Speed limit (30 km/h)							|
| .1641					| Turn left ahead								|
| .1300	      			| Keep right					 				|
| .0668				    | Speed limit (50 km/h)							|

Image #4: Turn right ahead

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000       			| Turn right ahead								| 
| .0000    				| Ahead only									|
| .0000					| Priority road									|
| .0000	      			| Go straight or right			 				|
| .0000				    | Stop					     					|

Image #4: Road work

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000       			| Road work										| 
| .0000    				| Beware of ice/snow							|
| .0000					| Bicycles crossing								|
| .0000	      			| Slippery road			 						|
| .0000				    | Wild animals crossing		   					|


![alt text][image7] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here's an example of how the first convolutional layer activates edges:

![alt text][image8] 

