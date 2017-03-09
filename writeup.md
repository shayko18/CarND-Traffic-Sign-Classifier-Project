#**Traffic Sign Recognition** 

##Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sign_example.png "Sign example"
[image2]: ./hist_before.png "Hist before"
[image3]: ./hist_after.png "Hist after"
[image4]: ./all_signs.jpg "All signs"
[image5]: ./my_signs.png "My signs"
[image6]: ./vis_cnn_1.png "vis cnn 1"
[image7]: ./vis_cnn_2.png "vis cnn 2"

---
###Writeup / README

####1. Writeup
You're reading it! and here is a link to my [project code](https://github.com/shayko18/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set and 
The code for this step is contained in the 2ed code cell of the IPython notebook. 
Because I saw that the validation set was only about 11% from the total inputs (train+validation) I coded two options to select the training set and the validation set:

- change_valid_size_en=0: Use the original training set and validation set. validation set is 11%


- change_valid_size_en=1: Use a validation set of 20%. I split the training data into a training set and validation set
 
The default is to use the given validation set (I understood that this is what you wanted)

At the end, regardless of the option we choose, we shuffle the data.

* The size of training set is 34799
* The size of validation set is 4410 (~11%)
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset
In the 3ed code cell we have some "help" functions that we will use in this work:

- preprocess: We have two stages here. The first is to convert the images to grayscale because in most of the images there is not much information on the color, and that will simplfy the CNN. In the second stage we normalize the image by subtracting the mean and dividing it by the difference between the max and the min values, This in order to have all images in the same position and contrast more or less

- get_fake_data: we are given a gray image and we "jitter" it a little bit by rotating, zooming and shifting it by a small random values
- plot_histo: we plot an histogram on a given vector


In the 4th code cell we simply plot one image from the training set with its sign index just to get some feel for the problem we are dealing with.
We see an example of the original image, the image after preprocessing and a "fake" image from this preprocessed image. Here is the random sign we plot (we got "No Entry"):
 
![alt text][image1]


###Design and Test a Model Architecture

####1. Preprocessing

The 5th code cell of the IPython notebook we are dealing with three issues:
- Looking at the distribution of the training data and plotting its histogram. An important question I asked myself is if we want to assume that the input data class is a random variable (has a pdf). If it does we can use the input data to try to estimate this pdf and use it to modify the cross entropy function (we use Pr(y|X)*Pr(X) and not just Pr(y|X)). I think it is logical to assume that we can say something about the a-priori distribution of the traffic signs: for example there are surly more 'Yield' signs than 'Wild animals crossing' signs. But I left this question open and coded the two options: one that assume an a-priori distribution (use_a_priory_dist=1, this is the default) and the seconed does not (use_a_priory_dist=0)
- Balancing the training data: We see from the histogram that there are class that are poorly represented compare to other class (class 0 Vs class 1 for example). We would like to train on a balanced data so the CNN we train on all the input class. We gave the option to balance the data (balance_en=1, this is the default). What we did is to make sure each class will have at least "class_minimal_val" samples. 
"class_minimal_val" is set to be the value of the average+std of the original samples. For each class we calculate by what factor we have multiply each sample from this class. Each new sample is also jittered from the original sample (rotation+zoom+shift)
- Generating "fake data": we wanted to give the option to increase the number of input samples we have (add_fake_data_en=1, this is the default). This will give the CNN more data to work on and should result in a better error rate. We increased the samples by a factor of 2 (configurable). Each new sample is also jittered from the original sample (rotation+zoom+shift). An example of the "fake data" was given before.


The difference between the original data set and the final data set we will work on is the following:


- Original number of training examples = 34799


- Number of training examples after balancing = 77397


- Number of training examples after generating fake data = 154794 

Here is the classes histogram of the original data, and the data after we balanced and added "fake data":

Before:

![alt text][image2]

After:

![alt text][image3]


####2. Model Architecture

The code for my final model is located in the 6th cell of the ipython notebook.

I started form the LeNet that was given to us. This architecture was good to identify digits, so we can assume it would be a good starting point.
After reading the paper you pointed us to, I added a connection from the output of the first layer to the flattening stage (via max pool of 2x2). I than saw that the training error was very close to zero so I thought that I was overfitting and that is why I added a dropout after the last two fully connected stages.

From the paper:
"The motivation for combining representation from multiple stages in the classifier is to provide different scales of receptive fields to the classifier.
In the case of 2 stages of features, the second stage extracts “global” and invariant shapes and structures, while the first stage extracts “local” motifs with more precise details"

So this is what we have:

	Input - 32x32x1
	
	stage 1: 

		conv of 5x5 with 1x1 stride, valid padding, outputs is 28x28x6

		activation with relu

		max pool 2x2 stride,  outputs 14x14x6
	
	stage 2: 

		conv of 5x5 with 1x1 stride, valid padding, outputs is 10x10x16

		activation with relu

		max pool 2x2 stride,  outputs 5x5x16

	stage 3:
		
		we take stage1 output and max pool with 2x2 stride. output is 7x7x6. flatten this output and concatenate with the flatten of the output of satge2. We get a vector of size 696

	stage 4: 

		Fully Connected. Input = 400+294. Output = 120

	stage 5: 

		Fully Connected. Input = 120. Output = 84

	Output - 43 classes
	
		



####3. Model Training.
In the 8th code cell there is the pipeline

I run the model for 15 epochs with batch size of 128.
In the first 10 epochs we run with learning rate of 0.001, in the last 5 epochs we lower the learning rate to 0.0001 in order to fine tune the model estimations.

We used the Adam optimizer.

Pay attention that if we choose to use the a-priori distribution we insert it here (if we choose not to use the a-priori distribution, the relevant vector will be a zeros vector). Explanation on it is in the previous sections

One more point: I think that for the real problem it would be a good idea to not weight all the errors with the same weight. For example if you are mistaken between 'stop' sign and 'No Entry' sign it is a "big" error. But if you are mistaken between 'Speed limit (20km/h)' to 'Speed limit (30km/h)' it is not that important (Anyway in Germany the speed limit is not relevant :-)) 

The hyperparameters are set in the 7th cell of the ipython notebook. 
I added 5 more epochs, that we use a finer learning rate factor (0.0001) so we will fine tune the CNN.
The rest of the parameters stayed as is.


####4. Solution Approach. 

The code for calculating the accuracy of the model is located in the 9th and 10th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.956 
* test set accuracy of 0.934

We can see that there is some difference between the test and validation results. this can be explained by the relatively small validation set. 
The approach to reach this CNN is described  before, we see now that we don't overfit (the test error is not zero). 
We can see the benefit in the "gear shifting", the error drops when we start to use the finer gear (0.0001)


###Test a Model on New Images

####1. Acquiring New Images

Here are all the 43 traffic signs:
![alt text][image4]

in the 11th cell we plot the six German traffic signs that I found on the web:
Sign_0) Index=#4 : Speed limit (70km/h)
Sign_1) Index=#9 : No passing
Sign_2) Index=#11: Right-of-way at the next intersection
Sign_3) Index=#12: Priority road
Sign_4) Index=#14: Stop
Sign_5) Index=#32: End of all speed and passing limits

![alt text][image5]

We expect that round signs will "compete" with round signs. 
The same goes for the triangle signs (warning signs).
There are 4 signs with diagonal lines on them - we can expect they will also "compete".
In other words we can expect that the errors (or the signs that comes as the next most likely) will be inside the "round family" or inside the "triangle family". For example 'End of no passing by vehicles over 3.5 metric tons' to 'End of no passing'. 


####2. Performance on New Images

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

Here are the results of the prediction:

	(4)  Speed limit (70km/h)                  --> predict: Speed limit (20km/h)
	(9)  No Passing                            --> predict: No Passing 
	(11) Right-of-way at the next intersection --> predict: Right-of-way at the next intersection
	(12) Priority road                         --> predict: Priority road
	(14) Stop                                  --> predict: Stop
	(32) End of all speed and passing limits   --> predict: End of all speed and passing limits


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. This is less than the accuracy on the test set of ~94%.


####3. Model Certainty - Softmax Probabilities

The code for making predictions on my final model is also located in the 12th cell of the Ipython notebook.

5 out of 6 images I selected were predicted correctly.
 - Four prediction were vey good (>99.9% likleyhood)
 - One prediction with 99.4% likleyhood
 - One error



Also here (like I explained before) we can use or not use the a-priori probabilty of the signs. As before, I used this probability.

Here are the results for the 6 images we looked on:

Softmax For Speed limit (70km/h):

      Sign=Speed limit (20km/h)                                 with Pr=9.52e-01
      Sign=Speed limit (70km/h)                                 with Pr=4.78e-02
      Sign=Road narrows on the right                            with Pr=5.19e-06
      Sign=Keep left                                            with Pr=4.49e-06
      Sign=Traffic signals                                      with Pr=3.29e-06
	
	We see that most of the most likely options has circles in them, and the sign itself is round. The number 20 is similar to 70 (regarding the shape of them), so it is logocal to imagine that that kind of error could happen. The right sign was 2ed most likley here.

Softmax For No Passing:

      Sign=No passing                                           with Pr=1.00e+00
      Sign=End of no passing                                    with Pr=2.85e-06
      Sign=Stop                                                 with Pr=2.37e-09
      Sign=End of all speed and passing limits                  with Pr=8.05e-10
      Sign=No entry                                             with Pr=1.44e-11
	
	We see that "End of no passing" is very unlikely, but still in the top 5. we see that all the signs are round signs

Softmax Right-of-way at the next intersection:

      Sign=Right-of-way at the next intersection                with Pr=1.00e+00
      Sign=Beware of ice/snow                                   with Pr=7.36e-07
      Sign=Children crossing                                    with Pr=1.95e-11
      Sign=Double curve                                         with Pr=1.38e-11
      Sign=Slippery road                                        with Pr=5.21e-14
	   
	We see that all the signs are warning sign: triangle signs 

Softmax For Priority road:

      Sign=Priority road                                        with Pr=1.00e+00
      Sign=Roundabout mandatory                                 with Pr=2.96e-07
      Sign=End of no passing by vehicles over 3.5 metric tons   with Pr=7.49e-08
      Sign=Right-of-way at the next intersection                with Pr=5.27e-09
      Sign=End of no passing                                    with Pr=6.04e-10

	"Priority road" has a special shape, we see that is the top 5 there are round signs 

Softmax For Stop:

      Sign=Stop                                                 with Pr=9.94e-01
      Sign=Speed limit (70km/h)                                 with Pr=5.70e-03
      Sign=Speed limit (20km/h)                                 with Pr=1.92e-04
      Sign=Speed limit (30km/h)                                 with Pr=7.78e-06
      Sign=Keep left                                            with Pr=1.79e-06
	
	This was the a little more difficult prediction. the shape is close to being round, so round signs were selected

Softmax For End of all speed and passing limits:

      Sign=End of all speed and passing limits                  with Pr=1.00e+00
      Sign=End of speed limit (80km/h)                          with Pr=2.70e-04
      Sign=End of no passing                                    with Pr=1.39e-05
      Sign=Children crossing                                    with Pr=2.67e-06
      Sign=Speed limit (30km/h)                                 with Pr=7.97e-07
	
	we see that this sign has the diagonal lines like the sign "End of no passing". We expected this. In other runs the other signs with diagonal line were in the top 5 likley signs here.
 

###Visualize the Neural Network's State with Test Images
We plotted the visual output after the first two satges of the "Speed limit (70km/h)" sign. We can see nicly after the first stage clear characteristics, for example the circle and the "70".
From the 2ed stage it is more difficult to see somthing clear.

Here is what we got in the 1st stage:
![alt text][image6]

Here is what we got in the 2st stage:
![alt text][image7]
