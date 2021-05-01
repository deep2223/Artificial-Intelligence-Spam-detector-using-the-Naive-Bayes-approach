# Spam-detector-using-the-Naive-Bayes-approach

libraries used in project : NumPy, math, re, sys and Matplotlib.

1. **spam_detector.py** :
This file contains all the functions related to build a probabilistic model from the training set for SPAM detector (which will create model.txt file) plus all the functions related to test the model on given test dataset. This file will create a single file called result.txt with classification results wether it's SPAM or HAM. 

2. **spam_detector.ipynb** :
As an alternative to 'spam_detector.py'. Has the same code and functionality as 'spam_detector.py'. In case, execution of this file is needed, use jupyter notebook to run it.

2. **model.txt** :
The resulting model from spam_detector.py will be saved in this file as per given format in project description.

3. **result.txt** :
For each test file, the classification result will be saved in this file.

Note: Please go through the submitted report for detailed analysis of classifier.

To run the code:
Paste both train and test dataset folders where the spam_detector.py is located. 

1. **With command line**
With terminal, go to the folder where the spam_detector.py is located and run below command:

    python spam_detector.py

This will generate both model.txt and result.txt.

2. **With pycharm IDE**
Open the 'spam_detector.py' file in IDE. Paste train and test dataset folder with the file. Run the spam_detector.py. It'll generate model.txt and result.txt.

3. **With jupyter notebook**
Open the 'spam_detector.ipynb' file in jupyter notebook. Paste train and test dataset folder with the file. Run the spam_detector.ipynb. It'll generate model.txt and result.txt.

PLEASE NOTE : As output, a window will pop-up showing the confusion matrix for ham classification. In case the confusion matrix is not fully visible, please resize the window. To proceed with other outputs, please close the current window and it will lead you to the next window.
