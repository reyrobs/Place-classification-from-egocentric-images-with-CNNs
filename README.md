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
Our original data set comprises of 97909 images of various locations which will represent our 24 distinct classes, taken from wearable devices throughout the world. We have included a full depiction of the data set as well as the percentage of each label in the table on the next page along with a bar plot (please refer to page 4) to better visualise the diversity of the data. The bar plot clearly indicates that the data set is highly unbalanced, whereby \textit{office} occupying a much larger proportion of the labels than the others. Referring to our tables, we can see that the \textit{office} label makes up for around 40\% of the data with about 36,000 entries, while the rest accounts for varying proportions. Additionally, we have included a collage comprising of an image from each label as a reference on what the images we will be manipulating and feeding to our model resemble, this is simply to give a better idea of the quality and image type of each image from the data set that we will be dealing with.

### Bar Plot

### Collage of image from each class

<p align="right">(<a href="#top">back to top</a>)</p>

## Results with imbalanced dataset

![alt text](https://github.com/reyrobs/Bank-Customer-Prediction/blob/main/images/results_imbalanced.png?raw=true)
<br>
The results obtained on this dataset are not as good as they seem. If we look at the overall accuracy obtained, we see that it seems rather good. However since the data is unbalanced, we see rather poor metrics obtained for the label 0, for precision, recall and f1-score. The classifier has developped a tendency to predict for label 0 since it represents a larger proportion of the labels. 

<p align="right">(<a href="#top">back to top</a>)</p>


## Results with balanced dataset

### Method 1: Undersampling
![alt text](https://github.com/reyrobs/Bank-Customer-Prediction/blob/main/images/under_sampling.png?raw=true)
<br>
Our first method to combat the unbalanced dataset gives us better results for the metrics of class 1. Although the overall accuracy has decreased, it now represents a better representation of its true value since the dataset is now balanced.  The way that this method works it rather simple, whereby we simply 
remove the number of elements from the overpopulated label. The downside of this is that we are throwing away data that could otherwise be used for our classification. 
### Method 2: Oversampling
![alt text](https://github.com/reyrobs/Bank-Customer-Prediction/blob/main/images/over_sampling.png?raw=true)
<br>
The second method used is oversampling, which is very similar to the first method except that we increase the samples of the underrepresented class by creating duplicated samples. We can see that we obtain good metrics for each class as well as a solid accuracy.

### Method 3: SMOTE

![alt text](https://github.com/reyrobs/Bank-Customer-Prediction/blob/main/images/smote.png?raw=true)
<br>
Our last and final method is the smote method (synthetic minority oversampling technique) which essentially creates artifical samples for the underrepresented class such that the dataset now becomes balanced. This was the method which yielded the best results for both the metrics of each class as well as the overall accuracy. 
<p align="right">(<a href="#top">back to top</a>)</p>

### Confusion matrix for best method
![alt text](https://github.com/reyrobs/Bank-Customer-Prediction/blob/main/images/confusion_matrix.png?raw=true)

## Interpretation of results
Over the course of this small project we have seen the effect of using an unbalanced dataset. This can gives a misleading accuracy since the classifier will develop a tendency to classify the overrepresented class. In order to tackle this, we have made use of 3 methods and have found that the best one in this case was the SMOTE method. 
<p align="right">(<a href="#top">back to top</a>)</p>


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
Bank Dataset Kaggle,
https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling
