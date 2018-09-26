# **Traffic Sign Recognition** 

## Writeup/README

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the [data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/labels_distribution.jpg "Visualization"
[image2]: ./results/comparison.jpg "Grayscaling"
[image4]: ./sources/sign1.jpg "Traffic Sign 1"
[image5]: ./sources/sign2.jpg "Traffic Sign 2"
[image6]: ./sources/sign3.jpg "Traffic Sign 3"
[image7]: ./sources/sign4.jpg "Traffic Sign 4"
[image8]: ./sources/sign5.jpg "Traffic Sign 5"
[image14]: ./results/sign1.jpg "Traffic Sign 1"
[image15]: ./results/sign2.jpg "Traffic Sign 2"
[image16]: ./results/sign3.jpg "Traffic Sign 3"
[image17]: ./results/sign4.jpg "Traffic Sign 4"
[image18]: ./results/sign5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

You are reading my Writeup/README. 
Here is a link to my [project code](https://github.com/OctopusNO1/Classify-TrafficSign/blob/master/Traffic_Sign_Classifier.ipynb)
Here is a link to my [report.html](https://github.com/OctopusNO1/Classify-TrafficSign/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used "shape" and "len()" to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessed the image data.  

As a first step, I decided to convert the images to grayscale because this task is color independent, grayscale can decrease the computations and train time.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

#### 2. Model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 12x12x40 	|
| RELU					|												|
| Dropout			    | 50% keep chance								|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 10x10x80 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x80 				    |
| Flatten   	      	| outputs 2000                                 |
| Fully connected		| outputs 120	      						    |
| RELU					|												|
| Fully connected		| outputs 84	      						    |
| RELU					|												|
| Fully connected		| outputs 43	      						    |
|						|												|

#### 3. Trained model

To train the model, I used 

| Hyperparameter       |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| epochs      			| 40											|
| batch size			| 128											|
| dropout probabilities	| 50%											|
| learning rate			| 0.001, 0.0001									|
| optimizer     		| Adam      	      						    |

#### 4. results and discuss 

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.964
* test set accuracy of 0.948

I choose a well known architecture "LeNet". I add "dropout layer" to prevent overfit.
 

### Test a Model on New Images

#### 1. Test five new German traffic signs

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The fourth image might be difficult to classify because it has a different shape and some watermarks.  

#### 2. Predictions results on these five images

Here are the results of the prediction:

| Image			                                |     Prediction	        					| 
|:---------------------------------------------:|:-------------------------------------------:| 
| Speed limit (30km/h)-1                       | Speed limit (30km/h)-1   					    | 
| No entry-17    			                    | No entry-17 									|
| Road work-25					                | Road work-25									|
| Road work-25	      		                    | Keep right-38					 				|
| Right-of-way at the next intersection-11     | Right-of-way at the next intersection-11     |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.  

Why was the fourth image predicted to 25? We can see the distribution plot of the data above. 
The data with 25 label is more than others, it may lead biased toward the classes with more images. 
We can use data augmentation to increase the less images and balance the distribution to decrease the bias.

#### 3. Top 5 softmax probabilities for each image

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the correct predicted image, the model is very sure. For the fourth image, the model is not very sure.  
The top five soft max probabilities were

![alt text][image14]  
![alt text][image15]  
![alt text][image16] 
![alt text][image17]  
![alt text][image18]



