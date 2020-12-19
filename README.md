# FeedNet - A YOLO Model for Grocery Object Detection

## Abstract


## Problem Statment
The Covid-19 Pandemic has created economic hardship and problems in the food supply chain, causing the demand for charitable food assistance to dramatically increas. However, Covid-19 has also impacted the ability for food banks to meet these demands, because they mainly rely on in person volunteers. Volunteers are most likely to do manual tasks such as sorting food and stocking shelves. Because in person volunteering is dangerous during the pandemic, FeedNet is an investigation into the feasability of using computer vision for identifying food items. This investigation aims to prototype a model that could be used in a robot that can assist in food bank tasks.

## Related Work
The related works that I consulted in this investigation are listed below with brief descriptions:
1. Background information on food banks and the Covid crisis: https://www.feedingamerica.org/sites/default/files/2020-10/Brief_Local%20Impact_10.2020_0.pdf
2. Paper on classification with Frieburg Dataset: http://ais.informatik.uni-freiburg.de/publications/papers/jund16groceries.pdf
3. Frieburg Dataset annotated with object detection labels: https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset
4. Information on training with Darknet in Colab: https://medium.com/@quangnhatnguyenle/how-to-train-yolov3-on-google-colab-to-detect-custom-objects-e-g-gun-detection-d3a1ee43eda1
5. Paper on YOLOv3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
6. Article on converting data with Roboflow: https://blog.roboflow.com/how-to-convert-annotations-from-pascal-voc-to-yolo-darknet/
7. pjreddie version of darknet: https://github.com/pjreddie/darknet
8. pjreddie Darknet YOLOv3 tutorial: https://pjreddie.com/darknet/yolo/
9. AlexeyAB version of darknet: https://github.com/AlexeyAB/darknet

## Methodology
The approach that I used for solving this problem was to use a YOLOv3 model and train on an object detection dataset of images of Groceries. 

### The Dataset
The Frieburg dataset is the dataset that I used, and it contains 4947 total images across 25 classes of common food items that were found in grocery stores (3). Each class has at least 97 images, each with several object instances. Because this dataset was originally used for classification, each image only has one class of object in it. The Frieburg Object Detection dataset that is linked above was originally in Pascal VOC format, so to convert it to YOLOv3 format using Roboflow (6). Roboflow has a different version of YOLOv3, than the pjreddie/darknet version that I used, so I wrote the bash script ```reorganization.sh``` to make the dataset compatible with the pjreddie/darknet version of YOLOv3. You can also download the YOLOv3 version of the data from my website with ```wget https://students.washington.edu/bhimar/frieburg.tar.gz```.

### The Model
Because in practical applications, this network would have to be very fast for use in real time object detection, I chose to use a YOLOv3 model. The Frieburg dataset is relatively small, I so initialized the network with YOLOv3 weights trained on ImageNet which was provided by pjreddie (8). This would hopefully improve our model and decrease our training time. Because I do not have an NVIDIA GPU in my laptop, I trained the network on Google Colab. The colab notebook that I wrote for training is ```FeedNet_YOLOv3_Darknet.ipynb```, and I adapted code from the Quang Nguyen for training in Colab (4). I trained for 6000 iterations, which took around 13 hours on the Google Colab GPU.

## Experiment and Evaluation
I am evaluating my results using mean average precision (mAP), and the results are shown below. To do this, I had to use the AlexeyAB fork of darknet, which entailed some manual reorganization of the data which is not included in this repo (9). Also, I evaluated my results qualitatively by inspecting images from inference on the test set and inference on a video that I took with my phone of food items in my house. The inference was done in another Colab notebook called ```FeedNet_Inference.ipynb```, and I also adapted code from Quang Nguyen for this part as well (4).

## Results
### mAP Analysis
Coming soon! (currently running)

### Qualitative Analysis
The qualitative analysis that I am presenting references the examples below. We see that in the inference examples for the test set, the model actually performs pretty well! The model is able to identify several object correctly such as corn, tea, coffee, vinegar, chocoalte, milk, pasta, and beans. The model incorrectly indentifies an instance of pasta and soda and vinegar. In the video inference example, we see that the model doesn't perform great in a more natural setting. It is able to correctly identify coffee most of the time, and the tea, beans and vinegar some of the time. The ramen was not in the dataset, but I thought it might be close enough to pasta for the model to identify. It is able to identify the object instance, but the model has trouble reliably classifying them. The model thinks that the tea and ramen are candy a lot of the time, perhaps because of the packaging and color. The model also thinks that the vinegar is either juice or water sometimes, perhaps because of the shape of the bottle.

### Challenges and Takeaways
We see that the model has good performance on the testing set, but poor performance on the natural video. This task and dataset is a very difficult one because of the diversity of food packaging that can be found in a grocery store. Because these are packaged foods, the model is trying to predict the kind of food based on the packaging, not the food itself. For this reason, it has to rely on detecting features in the labels and containers. From the Frieburg paper, I already knew that this was going to be a challenging task (2) due to the variety of images, angles, lighting conditions, and clutter. I think that the most significant obstacle for this task is the variety of packaging for the food. The model is able to identify coffee, milk, and water quite well because generally the packaging across brands and products looks quite similar. However, for items such as pasta, beans, and candy, the packaging can vary a lot. It is possible that by collecting more examples and training longer that the model's performance can be improved. However, I think that likely an option for deploying the model for practical purposes would be to figure out the most common foods that food banks need to sort, and then decrease the number of classes used for the model. Even though likely a robot with these capabilities may only be able to sort some of the food objects, it would decrease the amount of sorting needed to be done by human volunteers. There is a lot more work to be done, and I am excited to continue exploring this!

## Examples
The following examples are from the inference Colab notebook. These are 10 randomly sampled images from the test set and used for inference. Above each are the classes for the objects.
![alt text](writeup/predictions.png?raw=true)

I also ran inference over a video of food items that I found in my house, so see how the model would perform in a more natural setting. The video inference was done with OpenCV, and the code for this is also in the inference Colab notebook. Because my laptop does not have an NVIDIA GPU, I was not able to do a real time video demo. The full version is at https://youtu.be/UgWAbjeQLRE, but I have included a short gif below.
![alt text](writeup/demo.gif?raw=true)