<div align="center">
<div id="top"></div>
<h2 align="center">Place-classification-from-egocentric-images-with-CNNs</h3>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#genereal-idea-and-approach">Genereal idea and approach</a></li>
    <li><a href="#data-exploration">Data Exploration</a></li>
    <li><a href="#results-with-imbalanced-dataset">Results with imbalanced dataset</a></li>
    <li><a href="#results-with-balanced-dataset">Results with balanced dataset</a></li>
    <li><a href="#interpretation-of-results">Interpretation of results</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Introduction to problem
The purpose of this experiment is to see whether the use of a CNN is appropriate for image classification of a dataset with initially 24 labels, which have been later reduced to 5 given that the dataset was very unbalanced. The CNN used for this was ResNet50, along with a transfer learning approach. We also make use of a heat map to observe which parts of the image have been used to make the classification. All of this and more are covered in the following sections. 

<p align="right">(<a href="#top">back to top</a>)</p>

## Data Exploration
Our original data set comprises of 97909 images of various locations which will represent our 24 distinct classes, taken from wearable devices throughout the world. We have included a full depiction of the data set with a bar plot to better visualise the diversity of the data. The bar plot clearly indicates that the data set is highly unbalanced, whereby the *office* class occupying a much larger proportion of the labels than the others. Additionally, we have included a collage comprising of an image from each label as a reference on what the images we will be manipulating and feeding to our model resemble, this is simply to give a better idea of the quality and image type of each image from the data set that we will be dealing with.

### Bar Plot
<!-- ![alt text](https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/barPlot.png?=250x250) -->
<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/barPlot.png?" width="400">


### Collage of image from each class
<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/all.png?" width="300">

<p align="right">(<a href="#top">back to top</a>)</p>

## How CNNs work

Convolutional Neural Networks (CNNs) are a special type of Neural Networks designed to be used mainly with 2-D image data, however they can also be used with 1-D or 3-D data. The main aspect of the network are the convolutional layers, from which the network's name is derived and they perform an operation called *convolution*. When we talk about a convolution operation in the context of CNNs, we are referring to a linear operation between the input data and a 2-D filter (also known as kernel), composed of weights designed to detect specific features from the input data. The filter is much smaller than the input data and the type of multiplication between the filter and the input data is an element wise dot product operation, to which the resulting elements are then summed up in order to give a single value as final result. The filter is then moved over a different area of the input data and the same procedure is applied, until the whole input data has been covered by the filter [[1]](#1). Please refer to the figure below for a visual representation of the process:

<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/CNN_filter.png?" width="400">

## ResNet50 model and Transfer learning
After the success of AlexNet at the LSVRC2012 classi-
fication contest, deep Residual Networks have become
increasing popular over the last few years in computer
vision and deep learning fields. These networks are able
to train up to hundreds of layers and achieve peak per-
formance. One of these networks which has emerged since is the ResNet50 model which is the one we will be utilizing during our experiments. Please refer to the figure below for a break down of the network.

<img src="https://github.com/reyrobs/Place-classification-from-egocentric-images-with-CNNs/blob/main/images/ResNet_Over.png?" width="700">



<!-- CONTACT -->
## Contact

Robert Rey - [LinkedIn](https://www.linkedin.com/in/robert-rey-36689a103/)

Project Link: [Bank-Customer-Prediction](https://github.com/reyrobs/Bank-Customer-Prediction)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## References
<a id="1">[1]</a> 
J. Brownlee, “How do convolutional layers work in deep learning neural networks?.”
