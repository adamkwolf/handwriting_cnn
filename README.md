# Convolutional Neural Network for Handwriting Classification

### Implementation of the Convolutional Neural Network
The neural network was developed using Python with TensorFlow, which has a built-in conv2d network that performs the convolutions to generate the filters. There are 2 convolutional layers of 5x5 pixels, 2 pooling layers 2x2 pixels, and 2 fully connected layers which join the filters together to form our output nodes for classification. The 2 fully connected layers consist of 47 output tensors (or 62 for by_class dataset). In order to get the prediction, we use the softmax function on the fully connected layer. The neural network was trained with 100,000 iterations on a batch size of 64 training images, which took about 1 hour and 30 minutes to complete. The network was tested with 10,000 images, which resulted in a mean accuracy of 90%.

### Confusion Matrix of By_Merge testing dataset
The confusion matrix is from By_Merge testing dataset. We can see that the network was able to learn each class but there was some confusion among similar characters such as O and 0, L and 1, I and 1. Digits appeared to be tested the most so the testing dataset could be biased to numbers and thus the accuracy could be also biased, but it's hard to tell, another explanation is that there are a lot more letter than number classes which can cause less predictions for letters. Going forward, one way for the network to learn the difference between similar characters like O and 0 is to add context learning so it can learn the context in which the drawings come from. Such as if they are typing a word they are more likely to want a “o” instead of “0”.
![](https://github.com/adamkwolf/handwriting_cnn/blob/master/stats/confusion-40000.png "Confusion Matrix of By_Class testing dataset")

### Error Examples
These are testing samples that were incorrectly predict by the trained network on the by_merge dataset. As we can see, the True value and Predicted value are very similar compared to the images so the errors are completely justified in the network's decision. As mentioned above context training would solve many of these errors.
![](https://github.com/adamkwolf/handwriting_cnn/blob/master/stats/plot-40000.png "Error Examples")

### Server Implementation
A Django app was created that runs off of Python, JavaScript, and HTML. It also uses a Redis server which manages the websocket traffic between the frontend and backend. The user will be able to the draw a character, which will be down sampled to match the size of the 28x28 pixels images that were inputted into the neural network. Once a drawing is completed, the drawing is sent to the server. Then, it is run through the neural network, which results in a prediction. The prediction is returned to the front-end through the websocket and outputted into a text box. 

### Main File Structure
```
├── cnn_train_and_test.py  # main training and testing script
├── Dataset.py  # basic data structure for handling the dataset
├── info_proj
│   ├── hw_site  # app for connecting the site to the backend
│   ├── main_app
│   │   ├── cnn_eval.py  # cnn api for the frontend
│   │   ├── consumers.py  #  endpoint for the web socket
│   │   ├── model/  # contains trained models for by_merge, by_class, and balanced
│   │   ├── routing.py  # web socket routing
│   │   ├── settings.py  # django settings
│   │   ├── urls.py  # main index url
│   ├── manage.py  # django startup script 
│   ├── static
│   │   └── canvas.js  # main canvas served from django to the frontend
│   └── templates
│       └── hw_site
│           └── index.html  # main index page served to the frontend which contains the script
├── load_data.py  # script created to load the dataset into python
├── out  # output file from training
└── stats/  # output directory that contains the confusion matrix and incorrect sample images
```

### Dependencies
```
python3 or above
Tensorflow
Numpy
Scipy
Matplotlib (visualizations)
Django
redis-server (server websockets)
```
