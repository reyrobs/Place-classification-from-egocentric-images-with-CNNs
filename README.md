<div align="center">
<div id="top"></div>
<h2 align="center">Place-classification-from-egocentric-images-with-CNNs</h3>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction-to-problem">Introduction to problem</a></li>
    <li><a href="#data-exploration">Data Exploration</a></li>
    <li><a href="#how-cnns-work">How CNNs work</a></li>
    <li><a href="#resnet50-model">ResNet50 model</a></li>
    <li><a href="#transfer-learning">Transfer Learning</a></li>
    <li><a href="#preprocessing-the-initial-data">Preprocessing the initial data</a></li>
    <li><a href="#creating-our-model">Creating our model</a></li>
    <li><a href="#results-and-loss-curves">Results and loss curves</a></li>
    <li><a href="#results-on-test-set">Results on test set</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Introduction to problem
The purpose of this experiment is to see whether the use of a CNN is appropriate for image classification of a dataset with initially 24 labels, which have been later reduced to 5 given that the dataset was very unbalanced. The CNN used for this was ResNet50, along with a transfer learning approach. We also make use of a heat map to observe which parts of the image have been used to make the classification. All of this and more are covered in the following sections. 

<p align="right">(<a href="#top">back to top</a>)</p>

## Data Exploration
Our original data set comprises of 97909 images of various locations which will represent our 24 distinct classes, taken from wearable devices throughout the world. We have included a full depiction of the data set with a bar plot to better visualise the diversity of the data. The bar plot clearly indicates that the data set is highly unbalanced, whereby the *office* class occupying a much larger proportion of the labels than the others. Additionally, we have included a collage comprising of an image from each label as a reference on what the images we will be manipulating and feeding to our model resemble, this is simply to give a better idea of the quality and image type of each image from the data set that we will be dealing with.

Bar Plot             |  Collage of image from each class
:-------------------------:|:-------------------------:
<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/barPlot.png?" width="400">  |  <img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/all.png?" width="300">



## How CNNs work

Convolutional Neural Networks (CNNs) are a special type of Neural Networks designed to be used mainly with 2-D image data, however they can also be used with 1-D or 3-D data. The main aspect of the network are the convolutional layers, from which the network's name is derived and they perform an operation called *convolution*. When we talk about a convolution operation in the context of CNNs, we are referring to a linear operation between the input data and a 2-D filter (also known as kernel), composed of weights designed to detect specific features from the input data. The filter is much smaller than the input data and the type of multiplication between the filter and the input data is an element wise dot product operation, to which the resulting elements are then summed up in order to give a single value as final result. The filter is then moved over a different area of the input data and the same procedure is applied, until the whole input data has been covered by the filter [[1]](#1). Please refer to the figure below [[2]](#2) for a visual representation of the process:

<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/CNN_filter.png?" width="400">

<p align="right">(<a href="#top">back to top</a>)</p>

## ResNet50 model
After the success of AlexNet at the LSVRC2012 classi-
fication contest, deep Residual Networks have become
increasing popular over the last few years in computer
vision and deep learning fields. These networks are able
to train up to hundreds of layers and achieve peak per-
formance. One of these networks which has emerged since is the ResNet50 model which is the one we will be utilizing during our experiments. Please refer to the figure below [[3]](#3) for a break down of the network.

<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/ResNet_Over.png?" width="700">

<p align="right">(<a href="#top">back to top</a>)</p>

## Transfer Learning
We will be using a method called transfer learning along our ResNet50 model, this means that instead of training the model from scratch, we keep the weights of our loaded model, whereby leveraging the information learnt from previous data and apply it to our data. There exists various ways of applying transfer learning but we will be focusing on fine tuning which is a more refined version of transfer learning, through which we essentially retrain some of the remaining layers of the model instead of just replacing the final layer for classification. By doing this, we hope to be able to capture more specific features of our data as this is what the final layers of our model are designed for. Please refer to the figure below [[4]](#4) for a visual representation of transfer learning. 

<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/transferPic.jpg?" width="500">

<p align="right">(<a href="#top">back to top</a>)</p>

## Preprocessing the initial data
Before setting up our model, it is important to preprocess our images in order to select the appropriate images such that we can run our experiment successively. By referring to the bar plot again from above, we see that the dataset is highly unbalanced, to which we will have to manipulate our data in order to account for this. Additionally due to Colab limitations of only being able to run for a certain amount of hours at a time, as well as other pressing matters of finishing the experiments in due time, it was in our best interest to proceed with using only 5 labels out of the 24 initially available in order to speed up the overall process. The labels we have decided to use are *Kitchen*, *Bedroom*, *Sport fields*, *Living room*, *Restaurant, Bar*.
</br>
</br>
After having decided which label to select, it was important to deal with the unbalancedness of the data set. By looking at the label with the least amount of samples, namely *Sport fields* we went ahead and set a range of 500-600 per label. Additionally it was important to use images from each label which were spread out as possible in order to provide diversity, instead of using images taken successively, which we implemented by using integer division as the distance between each image. For example, if we had 9352 images for a certain class, we would do 9352//500 = 18, which would mean we have a distance of 18 between each image selected for training. After applying this procedure for each of the label, we obtained the following frequencies for each label:

</br>
</br>

<table>
<thead>
  <tr>
    <th>Label</th>
    <th>Frequency</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Kitchen</td>
    <td>599</td>
  </tr>
  <tr>
    <td>Bedroom</td>
    <td>543</td>
  </tr>
  <tr>
    <td>Sport Fields</td>
    <td>538</td>
  </tr>
  <tr>
    <td>Living Room</td>
    <td>524</td>
  </tr>
  <tr>
    <td>Restaurant, Bar</td>
    <td>521</td>
  </tr>
</tbody>
</table>

<p align="right">(<a href="#top">back to top</a>)</p>

## Creating our model

The next step was to create our resnet model by making use of the *tensorflow* library. Since we are performing transfer learning, we have had to specify two parameters when creating our model, namely *weights=imagenet* and *include_top=False*. By setting *weights* to *imagenet*, we are making use of the predefined weights of the model when it was trained on the ImageNet dataset, which contains more than 14 million images, this allows us to leverage the more general features learned from the ImageNet dataset, which will also be present in our dataset. Additionally we have frozen most of the layers from the network, and focused on training the later layers of the model such that we can focus on learning the more specific features relevant to our dataset. Freezing the layers also allows us to greatly reduce the training time of our model. Furthermore by setting *include_top* to *False* we are removing the last Dense layer of the original resnet model, which comprises of 1000 classes and applying our own, which is made up of 5 classes instead, along with a softmax activation function, which is preferred when performing multi-class classification.

<p align="right">(<a href="#top">back to top</a>)</p>

## Results and loss curves

### Results with cross validation

Cross validation is used in order to test the reliability of a model. One main problem of manually splitting our data set into a training and testing group, is that we might unknowingly be favouring a particular label, or have a lot of noise in the training group which the model will pick up as useful information, which is also referred to as overfitting.
<br>
<br>
In order to try and avoid these issues, we can use cross validation, whereby we decide before hand how many times we want to train and validate our model, each time with a different training and validation group, then the results of each iteration are combined to give the final performance score. There exists different variations of cross validation, but we will be focusing on K-fold cross validation in our experiment. We have set K to 5, this means that we will be having a training and validation group five different times, such that the groups are different each time. Additionally setting K to 5 will give us a training and validation ratio of 80/20.
<br>
<br>
We have ran each fold for 10 epochs, such as creating a new model for each fold, we have included the best validation accuracy and validation precision for each model during each fold, as well as its distance from the mean squared, our results can be found in the table below before we go further in depth as to how to interpret them:

<table>
<thead>
  <tr>
    <th></th>
    <th>Val Accuracy</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Model 1</td>
    <td>0.819</td>
  </tr>
  <tr>
    <td>Model 2</td>
    <td>0.779</td>
  </tr>
  <tr>
    <td>Model 3</td>
    <td>0.773</td>
  </tr>
  <tr>
    <td>Model 4</td>
    <td>0.798</td>
  </tr>
  <tr>
    <td>Model 5</td>
    <td>0.780</td>
  </tr>
  <tr>
    <td>Mean</td>
    <td>0.789</td>
  </tr>
  <tr>
    <td>Standard Deviation</td>
    <td>0.0169</td>
  </tr>
</tbody>
</table>


### Loss curves with best model

<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/chart.png?" width="400">

### Interpretation of results

Looking at the table with the cross validation results, we see that the validation accuracy of each model is around the 0.80 mark, or 80% in other words, which we can consider of adequate accuracy for the given data set. As previously said, it is important to check that overfitting doesn't occur, one of the reason we employ cross validation. In order to check for this we can see whether the accuracy of each fold are closely related, if they are then we can be confident that there is no overfitting happening. We can see that all the accuracies are in the range (0.773-0.819) which is a small range. Additionally we can make use of the standard deviation to see how much the accuracies vary from the mean, confirming that the best accuracy of each model in fact do not differ much from the mean.
<br>
<br>
From the loss curves obtained, we can observe that both the
train and validation loss curves follow a similar trend,
i.e they both decrease and appear to flatten themselves
slowly as the number of epochs increases, which is
a reassuring sign, there are however a few things to
note. The two curves do not merge towards a common
point, to that we could say that there could be a case of
overfitting of the data. We would ideally want to see
the two curves flattening towards a common point.
We must also add that this plot was obtained over
only 10 epochs, and therefore it is not a straightforward
interpretation, however we are confident that given
more time, such as to run the training over a larger
number of epochs, say 100, that both the training and
validation loss curves would come to stability and meet
at a common point. Given the number of epochs that
we ran the training for, we are satisfied with the loss
curves obtained. 


<p align="right">(<a href="#top">back to top</a>)</p>

## Results on test set

### Metrics scores

<table>
<thead>
  <tr>
    <th></th>
    <th>precision</th>
    <th>recall</th>
    <th>f1-score</th>
    <th>support</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>Bedroom</th>
    <td>0.72</td>
    <td>0.78</td>
    <td>0.75</td>
    <td>109</td>
  </tr>
  <tr>
    <th>Kitchen</th>
    <td>0.71</td>
    <td>0.75</td>
    <td>0.73</td>
    <td>120</td>
  </tr>
  <tr>
    <th>Living room</th>
    <td>0.68</td>
    <td>0.68</td>
    <td>0.68</td>
    <td>105</td>
  </tr>
  <tr>
    <th>Restaurant,Bar</th>
    <td>0.87</td>
    <td>0.79</td>
    <td>0.83</td>
    <td>104</td>
  </tr>
  <tr>
    <th>Sport fields</th>
    <td>0.96</td>
    <td>0.92</td>
    <td>0.94</td>
    <td>107</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>accuracy</th>
    <td></td>
    <td></td>
    <td>0.78</td>
    <td>545</td>
  </tr>
  <tr>
    <th>macro avg</th>
    <td>0.79</td>
    <td>0.78</td>
    <td>0.78</td>
    <td>545</td>
  </tr>
  <tr>
    <th>weighted avg</th>
    <td>0.79</td>
    <td>0.78</td>
    <td>0.78</td>
    <td>545</td>
  </tr>
</tbody>
</table>

From the classification report obtained above we can see that *Sport fields* performed the best in terms of scores obtained for each metric. We believe the reason for that is because *Sport fields* has very different attributes than the other classes, meaning that the model is more easily able to differentiate an image belonging to *Sport fields* than it is for the others classes. As an example an image containing a swimming pool can only belong to the *Sport fields* label, and therefore the model learns to directly attribute a swimming pool to *Sport fields*, the same idea goes for a tennis court for example. 
<br>
<br>
Now that we have established why *Sport fields* achieved much better results than the other classes, let us try and depict the results that we achieved for the other classes. The results of the remaining classes don't seem to differ too much from each other, although the class *Restaurant, Bar* performed better for precision than the others. In the case of *Bedroom* and *Living room*, the two classes share similar features and so we think that this makes it more difficult for the model to differentiate between the two, thus reducing the scores obtained for each of the metrics. As far as accuracy goes, we obtained a score of 0.78, which was expected given that we made use of cross validation during training.

### Confusion matrix

<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/confusion (1).png?" width="400">

Let us now go over the confusion matrix obtained above, represented with the help of a heatmap ranging from light green to dark green, whereby dark green portraying larger numbers. It is useful to note that the confusion matrix can also be related to the classification report obtained previously. First we see that the main diagonal of the matrix is of a darker color than the rest of it, which is a positive sign, this entails that each classes were predicted correctly more often than not. Starting with the class *Sport, fields* again, we have a score of 97 at the point where *Sport, fields* on both axis meet, while the rest of the points range between 1-5. This is a clear indication that the model did not have too much trouble at predicting correctly the images labelled with *Sport, fields* for the reason that we have discussed before, in other words it did not get confused with the other labels. 
<br>
<br>
Moving on to the class *Bedroom*, we observe that even though it was predicted correctly 87 times out of 109, the model confused it the most with the label *Kitchen*, which came as a surprise to us since we would expect the model to confuse it the most with the label *Living room*, closely in second place. Similarly for *Kitchen*, it was wrongly predicted the most with the label *Bedroom*. However the class *Restaurant, Bar* was wrongly predicted the most with the class *Kitchen*, which doesn't come off as a surprise to us since the two share a lot of similar features. 

### Class Activation Maps (CAM)

Our last section of the results will make use of Class Activation Maps (CAM). CAM is a powerful method applied in the of field Computer vision for classification tasks. It enables the users working on the classification to evaluate which section of an image contributed the most to a diagnostic when using a particular model. In other words, say we are trying to classify images of cats and dogs, and we feed an image of a dog to which the model successfully classifies it as a dog, through the use of the CAM tool, we can visualize which section of the image contributed the most to classifying it as a dog. This can be beneficial when trying to improve the accuracy of the model and deciding which layers need to be tweaked, or deciding if the preprocessing should be done differently [[5]](#5). We have included a few snapshots of how CAM was used to classify a certain image along with the original image. Please refer to them afterwards, before our interpretation of what is happening.

<!--
![alt text](https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/pool_orig.jpg?)
<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/pool_orig.jpg?" width="400">
<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/pool_CAM.png?" width="400">
&emsp;&emsp;&emsp;&emsp;
-->

Original Image|  CAM Image
:-------------------------:|:-------------------------:
<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/pool_orig.jpg?" width="300"> | <img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/pool_CAM.png?" width="350">

Referring to the images above, we have an image which belongs to the *Sport fields* class, which the model successfully classified. If we look at the CAM image, we see that the model greatly made use of the water, represented by the color yellow in order to come up with a classification.

<br>
<br>

Original Image|  CAM Image
:-------------------------:|:-------------------------:
<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/kitchen_orig.jpg?" width="300"> | <img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/kitchen_CAM.png?" width="350">

In the sample image used above, our model incorrectly classified the image as *Living room* when it belonged to the *Kitchen* class. By looking at the activation map more thoroughly, we can observe that the model focused on the center of the image, where the counter is present, in order to make a decision. It is possible that it interpreted the counter of the kitchen and what lies on top of it as a table belonging to the *Living room*. 

<br>
<br>

Original Image|  CAM Image
:-------------------------:|:-------------------------:
<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/living_orig.jpg?" width="300"> | <img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/living_CAM.png?" width="350">

Our final example can be seen above, on which the model correctly classified as *Living room*. When we analyze the CAM, we see that the model made use more of the right part of the original image, where the cup is located, in order to come up with a diagnostic. 

## Conclusion

Over the course of this project, we have learnt a lot of new and intriguing concepts, such as what is deep learning and deep learning networks, transfer learning, fine tuning, semi-supervised learning and more. By getting familiar with all these concepts, we have applied our knowledge to a data set created through the use of wearable devices. As our experiment progressed, we have had to adapt and adjust our approach as we didn't foresee a variety of obstacles in a way that we believed would give the most optimal results as well as being delivered in a timely manner. Due to the nature of deep learning networks and the time at which they take to train, we have decided to focus on only using 5 of the original 24 available labels such that we could try various implementations and procedures in order to have a better understanding of the different parameters used in a deep learning network. We believe the results we obtained after modifying and applying transfer learning on the resnet50 network are very satisfactory, with an accuracy of nearly 0.80 during our testing phase. 

<!-- CONTACT -->
## Contact

Robert Rey - [LinkedIn](https://www.linkedin.com/in/robert-rey-36689a103/)

Project Link: [Place-classification-from-egocentric-images-with-CNNs](https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/edit/main/README.md)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## References
<a id="1">[1]</a> 
J. Brownlee, ???How do convolutional layers work in deep learning neural networks?.???

<a id="2">[2]</a>
R. Robinson, ???An introduction to cnns and deep learning.???

<a id="3">[3]</a>
K. S. N. Z. e. a. Peng, J., ???Residual convolutional neural network for predicting response of transarterial chemoembolization in hepatocellular carcinoma from ct imaging.,???

<a id="4">[4]</a>
Z. Chen, J. Cen, and J. Xiong, ???Rolling bearing fault diagnosis using time-frequency analysis and deep transfer convolutional neural network,???

<a id="5">[5]</a>
V. Alto, ???Class activation maps in deep learning.???
