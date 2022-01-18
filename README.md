# Bodypart
Problem Statement
The problem pertains to classification of body parts (x-ray images) using an ML/DL model.
Dataset
The dataset contains two folders i.e. train folder and test folder. 
The classes in the dataset are 9 namely ankle, elbow, foot, hand, hip, knee, shoulder, spine and wrist.
The train images are 2277 in total. Ankle, foot, hands and hip classes contain 300 images each.  While wrist contains 299 images, Elbow contains 268 images, knee contains 172 images, shoulder contains 222 images and spine contains 116 images.
Solution
Google colab is used for the solution to this problem. The foremost thing that was done was loading the dataset to the Google drive. Afterwards, the Google drive is mounted on the Google colab.
Then relevant libraries are imported.
Model selection – Since the problem pertains to classification of images, hence we are bound to use convolution network for better accuracy. The foremost reason for using convolution neural networks is their ability to keep the spatial information intact. However, the accuracy will not be optimal using a conventional convolutional network hence the ideal solution to this problem is using transfer learning. Transfer learning refers to using a pertained network that is already trained on some dataset. There are different frameworks that offer different pertained models. However, Tensor flow is the most favorite library for this due to its large community support, and fast speed. Tensor flow offers a few pertained networks. We have chosen VGG16 due to its light weight and better accuracy, as the name suggests, it has 16 layers. The architecture of VGG16 is:

In this case we have used the convolution base of the VGG16 but changed the three dense classifier layers on top and replaced them with two dense layers one with 256 neurons and the other with 9 neurons.

Then we have set the directory of the train and the test folders. Afterwards we have made a generator that yields images from the Google drive that first normalizes the images and then set the target size as 150 * 150. 
Then the model is compiled with loss = categorical crossentropy , optimizer = adam, and metrics = accuracy. categorical_crossentropy is used because there are more than two classes. Then the model is trained on 5 epochs and then saved.
Then the training and validation accuracy and loss are plotted.
Then a test generator is made and the values are predicted against the test generator. 
Then confusion matrix is plotted against the actual and predicted values. 
Then sensitivity, specificity, precision, and overall accuracy is calculated. 
