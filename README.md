# Convolutional Neural Network for Handwriting Classification

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
