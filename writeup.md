#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

- Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sign_example.png "Sign example"
[image2]: ./hist_before.png "Hist before"
[image3]: ./hist_after.png "Hist after"
[image4]: ./md_fa.png "MissDetect FalseAlarm"
[image5]: ./all_signs.jpg "All signs"
[image6]: ./my_signs.png "My signs"
[image7]: ./vis_cnn_1.png "vis cnn 1"
[image8]: ./vis_cnn_2.png "vis cnn 2"

---


#### Writeup
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

- preprocess: We have two stages here. The first is to convert the images to grayscale because in most of the "close" images (like speed limit of 20 or 120) there is not much information on the color, and that will simplfy the CNN. In the second stage we normalize the image by subtracting the mean and dividing it by the difference between the max and the min values. This in order to have all images in the same color scale and contrast more or less.

- get_fake_data: we are given a gray image and we "jitter" it a little bit by rotating, zooming and shifting it by a small random values. rotate by a random degree between +-10 deg, zoom by a random factor between 0.95 to 1.05 and shifting by a random value between +-2 pixels in each axis. 
- plot_histo: we plot an histogram on a given vector and a given bins (will be the 43 classes). 


In the 4th code cell we simply plot one image from the training set with its sign index and name just to get some feel for the problem we are dealing with.
We see an example of the original image, the image after preprocessing and a "fake" image from this preprocessed image. Here is the random sign we plot (we got the sign "Right-of-way at the next intersection"):
 
![alt text][image1]


###Design and Test a Model Architecture

####1. Preprocessing

The 5th code cell of the IPython notebook we are dealing with three issues:

- Looking at the distribution of the training data and plotting its histogram. An important question I asked myself is if we want to assume that the input data class is a random variable (has a pdf). If it does we can use the input data to try to estimate this pdf and use it to modify the cross entropy function (we use Pr(y|X)*Pr(X) and not just Pr(y|X)). I think it is logical to assume that we can say something about the a-priori distribution of the traffic signs: for example there are surly more 'Yield' signs than 'Wild animals crossing' signs. But I left this question open and coded the two options: one that assume an a-priori distribution (use_a_priory_dist=1, this is the default) and the second does not (use_a_priory_dist=0). I got better results when I used the a-priori distribution. 
- Balancing the training data: We see from the histogram that there are class that are poorly represented compare to other class (class 0 Vs class 1 for example). We would like to train on a balanced data so the CNN we train on all the input class. We gave the option to balance the data (balance_en=1, this is the default). What we did is to make sure each class will have at least "class_minimal_val" samples. 
"class_minimal_val" is set to be the value of the average+std of the original samples. For each class we calculate by what factor we have multiply **each sample** from this class. Each new sample is also jittered from the original sample (rotation+zoom+shift)
- Generating "fake data": we wanted to give the option to increase the number of input samples we have (add_fake_data_en=1, this is the default). This will give the CNN more data to work on and should result in a better error rate. We understand that an important difference between the different inputs of the same class can be due to rotation, zomoming and shifting. Those are easy to create synthetically, therefore we increased the samples by a factor of 2 (configurable). Each new sample is jittered from the original sample (rotation+zoom+shift). An example of the "fake data" was given before in "Exploratory visualization of the dataset".


The difference between the original data set and the final data set we will work on is the following:


- Original number of training examples = 34799


- Number of training examples after balancing = 77397


- Number of training examples after generating fake data = 154794 

Here is the classes histogram of the original data, and the data after we balanced and added "fake data". Of course we used the original ("before") data to estimate the a-priori distribution of the inputs.  

Before:

![alt text][image2]

After:

![alt text][image3]


####2. Model Architecture

The code for my final model is located in the 6th cell of the ipython notebook.

I started form the LeNet that was given to us. This architecture was good to identify digits, so we can assume it would be a good starting point.
After reading the paper you pointed us to, I added a connection from the output of the first layer to the flattening stage (via max pool of 2x2) as it was suggested in the paper. The reason for that is to have both **global**” and invariant shapes and structures from the second stage and from the first stage also get “**local**” motifs with more precise details.    
I than saw that the training error was very close to zero, and the validation error was about 6%, so I thought that I was overfitting and that is why I added a dropout after the last two fully connected stages.


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

	combine stages 1,2:
		
		we take stage1 output and max pool with 2x2 stride. output is 7x7x6. flatten this output and concatenate with the flatten of the output of satge2. We get a vector of size 696

	stage 3: 

		Fully Connected. Input = 400+294. Output = 120
		activation with relu
		dropout of 0.8

	stage 4: 

		Fully Connected. Input = 120. Output = 84
		activation with relu
		dropout of 0.8
	
	stage 5: 

		Fully Connected. Input = 84. Output = 43

	Output - is the 43 classes
	
		



####3. Model Training.

1. The hyperparameters are set in the 7th cell of the ipython notebook:
I run the model for 15 epochs with batch size of 128.
In the first 10 epochs we run with learning rate of 0.001, in the last 5 epochs we lower the learning rate to 0.0001 in order to fine tune the model estimations.

2. In the 8th code cell there is the training pipeline
Pay attention that if we choose to use the a-priori distribution we insert it here (if we choose not to use the a-priori distribution, the relevant vector will be a zeros vector). Explanation on it is in the previous sections. We used the Adam optimizer in order to minimize the cross entropy.
In this cell there are also two placeholders (w_stg1, w_stg2) to help us answer question 9 (extra question). Also "TopKV2" is defined to help us with future question (Output Top 5 Softmax).

One more point: I think that for the real problem it would be a good idea to **not** weight all the errors with the same weight. For example if you are mistaken between 'stop' sign and 'No Entry' sign it is a "big" error. But if you are mistaken between 'Speed limit (20km/h)' to 'Speed limit (30km/h)' it is not that important (Anyway in Germany the speed limit is not relevant :-)). As We can expect (and we will see soon) the guys that designed the sign thought of this (also for humans) and tried in most cases to make similar meaning signs look similar also in the shape. For example, the warning signs are all triangle. We will see that usually errors happen between signs that look the similar.





####4. Solution Approach. 

The code for calculating the accuracy of the model is located in the 9th and 10th cell of the Ipython notebook.

In the 9th cell we have the evaluation pipeline that will give us the accuracy of the model. On top of the accuracy we also calculate, and plot, the "miss detect" (1-recall) and "false alarm" (1-precision) probability per class. We will use those in "step 3" to try to understand what sign is likely to be mistaken and what sign is likely to be wrongly estimated.
At the bottom of the 9th cell we have the function that will give us the 5 most likely signs per estimation. We will use it in "step 3".

My final model results were:

* training set accuracy of 0.991
* validation set accuracy of 0.953 
* test set accuracy of 0.933

Here is the "miss detect" (1-recall) and "false alarm" (1-precision) probability per class. We will use those probabilities we the six signs we look into in the next section:

![alt text][image4]

We can identify signs that are more difficult too predict correctly, usually those had less samples in the original train data, and because we used the a-priory probabilities their errors are less important to the overall accuracy (for example sign #0). Also we can see which signs are more likely to get wrongly predicted - sign #30 for example. Sign #30 is "Beware of ice/snow" and I think the reason for its high false alarm is that it is difficult to identify the "snowflake" in the middle of the sign, but the rest of the sign is similar to other "triangle" signs and it's a-priori is not low so it can be a good prediction for not so clear triangle signs.  

We can see that there is some difference between the test and validation results. this can be explained by the relatively small validation set. 
The approach to reach this CNN is described before, we see now that we don't overfit as much as before (the test error is not zero and is closer to the validation error). 
We can see the benefit in the "gear shifting", the error drops when we start to use the finer gear (0.0001). We also can see that the accuracy (both of the validation and of the training) stops from improving, so we can stop the training.




###Test a Model on New Images

####1. Acquiring New Images

Here are all the 43 traffic signs:


![alt text][image5]


in the 11th cell we plot the six German traffic signs that I found on the web:

- Sign_0) Index=#4 : Speed limit (70km/h)
- Sign_1) Index=#9 : No passing
- Sign_2) Index=#11: Right-of-way at the next intersection
- Sign_3) Index=#12: Priority road
- Sign_4) Index=#14: Stop
- Sign_5) Index=#32: End of all speed and passing limits

![alt text][image6]

We expect that round signs will "compete" with round signs. 
The same goes for the triangle signs (warning signs).
There are 4 signs with diagonal lines on them - we can expect they will also "compete".
In other words we can expect that the errors (or the signs that comes as the next most likely) will be to signs that looks closer to it, with high level similar characteristics. For example, I would assume that **given** an error on an input of a "round" sign the predicted sign will also be round - inside the "round family". Another example could be between 'End of no passing by vehicles over 3.5 metric tons' to 'End of no passing' (Both has diagonal lines). 


####2. Performance on New Images

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

Here are the results of the prediction:

	(4)  Speed limit (70km/h)                  --> predict (X): Speed limit (20km/h)    
	(9)  No Passing                            --> predict (V): No Passing 
	(11) Right-of-way at the next intersection --> predict (X): Beware of ice/snow  
	(12) Priority road                         --> predict (V): Priority road
	(14) Stop                                  --> predict (V): Stop
	(32) End of all speed and passing limits   --> predict (V): End of all speed and passing limits


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.67%. This is less than the accuracy on the test set of ~93%. The Error happened on the 1st and 3ed signs. Let's see what is the probability to get 2/6 errors if we assume the error probability for each sign is the same and equal to (1-0.93). This assumption is not accurate as we saw in the miss detect probabilities:
Pr(2/6 err) = 15 * 0.93^4 * 0.07^2 = 5.5%  


####3. Model Certainty - Softmax Probabilities

The code for making predictions on my final model is also located in the 12th cell of the Ipython notebook.

4 out of 6 images I selected were predicted correctly:

 - Two predictions were very good and correct: The second most likely was very unlikely: <1e-9
 - Two prediction was good and correct: The second most likely has a likelyhood of about 1e-3 or 1e-4
 - Two errors (the "Speed limit (70km/h)" sign, 1st picture and "Right-of-way at the next intersection", 3ed picture). 
	 - As we will see shortly in both of them the 2ed most likely option was the right answer.
	 - We will also see that the signs that were wrongly predicted as the correct signs ("Speed limit (20km/h)" and "Beware of ice/snow") have a high false alarm probability, which means they are likely as wrong estimations. We could go one step further and calculate the false alarm for each class as a function of the "input" sign.
	 - We can see that the sign "Right-of-way at the next intersection" has relatively high miss rate (11%), so it is more likely than others to be wrongly predicted. 


Also here (like I explained before) we can use or not use the a-priori probability of the signs. As before, I used this probability.

Here are the results for the 6 images we looked on. For each input sign I also printed its miss detect (MD) probability. For each estimation I also printed the false alarm (FA) probability (on top of its likely probability)

   Softmax For Sign: Speed limit (70km/h) ;   Global MD Pr=0.06

      Sign=Speed limit (20km/h)                                 with Pr=1.00e+00 ;   Global FA Pr=0.43
      Sign=Speed limit (70km/h)                                 with Pr=3.48e-04 ;   Global FA Pr=0.03
      Sign=Traffic signals                                      with Pr=7.05e-07 ;   Global FA Pr=0.13
      Sign=Speed limit (30km/h)                                 with Pr=1.18e-08 ;   Global FA Pr=0.04
      Sign=Go straight or left                                  with Pr=4.75e-09 ;   Global FA Pr=0.19
	
	Here we got the error. We see that most of the most likely options has circles in them, and the sign itself is round. The number 20 is similar to 70 (regarding the shape of them), so it is logical to imagine that that kind of error could happen. The right sign was 2ed most likely here. We also see that "Speed limit (20km/h)" has a high FA probability 

   Softmax For Sign: No passing ;   Global MD Pr=0.03

      Sign=No passing                                           with Pr=1.00e+00 ;   Global FA Pr=0.03
      Sign=Vehicles over 3.5 metric tons prohibited             with Pr=1.40e-10 ;   Global FA Pr=0.06
      Sign=No vehicles                                          with Pr=5.19e-17 ;   Global FA Pr=0.08
      Sign=No passing for vehicles over 3.5 metric tons         with Pr=3.66e-17 ;   Global FA Pr=0.00
      Sign=End of no passing                                    with Pr=8.30e-18 ;   Global FA Pr=0.06
	
	We see that the miss detect for this sign is low (0.03). We also see that all the top five options are very similar to the "No passing" sign (especially on grey sacle). We see that all the signs are round signs, like we would expect.

   Softmax For Sign: Right-of-way at the next intersection ;   Global MD Pr=0.11

      Sign=Beware of ice/snow                                   with Pr=9.99e-01 ;   Global FA Pr=0.33
      Sign=Right-of-way at the next intersection                with Pr=1.39e-03 ;   Global FA Pr=0.03
      Sign=Slippery road                                        with Pr=2.45e-08 ;   Global FA Pr=0.24
      Sign=Wild animals crossing                                with Pr=6.06e-09 ;   Global FA Pr=0.04
      Sign=Dangerous curve to the right                         with Pr=1.47e-09 ;   Global FA Pr=0.29
	   
	We see that the "input" sign has a high miss detect probability (11%) and also that the sign that we estimated has high false alarm probability (33%). The correct sign is in the second place. We see that all the signs are warning sign: triangle signs, like we would expect.

   Softmax For Sign: Priority road ;   Global MD Pr=0.03

      Sign=Priority road                                        with Pr=1.00e+00 ;   Global FA Pr=0.02
      Sign=Roundabout mandatory                                 with Pr=9.69e-12 ;   Global FA Pr=0.19
      Sign=Keep right                                           with Pr=1.84e-17 ;   Global FA Pr=0.03
      Sign=Speed limit (50km/h)                                 with Pr=3.37e-20 ;   Global FA Pr=0.02
      Sign=Speed limit (100km/h)                                with Pr=1.16e-21 ;   Global FA Pr=0.04

	"Priority road" has a special shape (diamond shape) that is closer to round than to triangle, we see that is the top 5 there are round signs. We can also see that both "Priority road" and "Roundabout mandatory" has an inner shape that is the same as the outer shape. like diamond inside a diamond, or a circle (three arrows that are almost like a circle) inside a circle.   

   Softmax For Sign: Stop ;   Global MD Pr=0.03

      Sign=Stop                                                 with Pr=9.99e-01 ;   Global FA Pr=0.05
      Sign=Speed limit (60km/h)                                 with Pr=1.09e-03 ;   Global FA Pr=0.13
      Sign=Speed limit (120km/h)                                with Pr=3.24e-05 ;   Global FA Pr=0.10
      Sign=End of all speed and passing limits                  with Pr=1.46e-05 ;   Global FA Pr=0.34
      Sign=Priority road                                        with Pr=1.58e-06 ;   Global FA Pr=0.02
	
	This was the a little more difficult prediction. The shape (octagon) is close to being round, so round signs were selected

   Softmax For Sign: End of all speed and passing limits ;   Global MD Pr=0.05

      Sign=End of all speed and passing limits                  with Pr=1.00e+00 ;   Global FA Pr=0.34
      Sign=End of no passing                                    with Pr=1.61e-04 ;   Global FA Pr=0.06
      Sign=End of no passing by vehicles over 3.5 metric tons   with Pr=7.04e-05 ;   Global FA Pr=0.04
      Sign=End of speed limit (80km/h)                          with Pr=1.38e-05 ;   Global FA Pr=0.05
      Sign=Roundabout mandatory                                 with Pr=6.56e-07 ;   Global FA Pr=0.19

	
	We see that this sign has the diagonal lines like the sign "End of no passing". We expected this. We expect signs with diagonal line were in the top 5 likeliest signs here.
 

###Visualize the Neural Network's State with Test Images
We plotted the visual output after the first two stages of the "Speed limit (70km/h)" sign. We can see nicely after the first stage clear characteristics, mainly edges in the image, for example the circle and the "70" edges.
From the 2ed stage it is more difficult to see something clear. we would expect to see some higher level features.

Here is what we got in the 1st stage:

![alt text][image7]

Here is what we got in the 2st stage:

![alt text][image8]
