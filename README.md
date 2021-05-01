# Spam-detector-using-the-Naive-Bayes-approach

Build a probabilistic model from the training set (available on Moodle). The code must parse the files in the training set and build a vocabulary with all the words it contains. Then, for each word, compute their frequencies and probabilities for each class (class ham and class spam).

To process the texts, fold all characters to lowercase, then tokenize them using the regular expression re.split(’\[\^a-zA-Z\]’,aString) and use the set of resulting words as your vocabulary.

For each word wi in the training set, save its frequency and its conditional probability for each class: P(wi|ham) and P(wi|spam). These probabilities must be smoothed using the ‘add δ’ method, with δ = 0.5. To avoid arithmetic underflow, work in log10 space.

Save model in a text file called model.txt. The format of this file must be the following: 1. A line counter i, followed by 2 spaces.

  1 abc 3 0.003 40 0.4 \
  2 airplane 3 0.003 40 0.4 \
  3 password 40 0.4 50 0.03 \
  4 zucchini 0.7 0.003 0 0.000001
  
Evaluating Model : Once implemented the classifier, use it to train a model that can classify emails into their most likely class: ham or spam. Run classifier on the test set given and create a single file called result.txt with your classification results. For each test file, result.txt must contain:

  1. a line counter, followed by 2 spaces
  2. the name of the test file, followed by 2 spaces
  3. the classification as given by your classifier (the label spam or ham), followed by 2 spaces 
  4. the score of the class ham as given by your classifier, followed by 2 spaces
  5. the score of the class spam as given by your classifier, followed by 2 spaces
  6. the correct classification of the file, followed by 2 spaces
  7. the label right or wrong (depending on the case), followed by a carriage return.

 For example, the result file could look like the following:

  1 test-ham-00001.txt ham 0.004 0.001 ham right 2 test-ham-00002.txt spam 0.002 0.03 ham wrong

libraries used in project : NumPy, math, re, sys and Matplotlib.

<br>

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
