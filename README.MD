***Czech language Diacritization restoring system report***


**Overview “diacritization.py”**
***How to run section is on bottom of the page.

The script “diacritization.py” involves several Python modules and utilizes the argparse library to handle command-line arguments, allowing it to switch between training and prediction modes. It uses datasets provided via URLs, downloading them if they are not present locally. The script is structured to support operations both with and without an internet connection, provided the necessary data files are available. All the datasets used in system for the training and predicting purposes was taken from NPFL129 “Machine Learning for Greenhorns” - Winter 22/23 class, taught by prof. Milan Straka.

**Data Preprocessing**

Dictionary and Dataset Loading:

The Dictionary class downloads a word variant dictionary if it does not exist locally. This dictionary likely maps non-diacritic words to possible diacritic forms.
The Dataset class downloads textual data for training and testing, reads it from a file, and creates two versions: one with diacritics (target) and one without (data), which is generated by translating characters using a predefined translation table.

Character Handling:

A translation table (DIA_TO_NODIA in the “Diacritization_Model”) usable with “str.translate” to rewrite characters with diacritics to the ones without them. This table is utilized to convert the training dataset text into a form without diacritics, preparing it for model input.

Model Creation

The “Diacritization_Model” class defines a model using the Multi-layer Perceptron (MLP) classifier from “Scikit-learn”. It is set up with specific parameters like hidden layer sizes, activation functions, and learning rates.
The model uses a “OneHotEncoder” for encoding characters, which helps in handling categorical character data efficiently.
The preprocessing method within the model prepares the input data by creating windows around each character which is in the “LETTER_NODIA” or “LETTER_DIA”, meaning that it has potential diacritic equivalent, encoding this contextual information using one-hot encoding. Then it one-hot encodes each character on that selected window, by their corresponding integer value from “letter_dict”. So we are having one-hot encoded data (2D array for each window, whose sizes depend on the “letter_dict”) which maps to the target integer which is the integer equivalent of that letter (which is either in LETTER_NODIA or LETTER_DIA as described above) from “letter_dict”.  This windowing helps in capturing the local context around each character, which is crucial for accurate diacritization.



**Prediction**

In prediction mode, the model loaded from a file predicts diacritics for input text. The prediction function handles each character which is in “LETTER_NODIA” (as the text for which the model should predict does not have any diacritics character) by considering its surrounding context (similar to the training phase), and the output is then mapped back to words with diacritics using the dictionary loaded earlier.
The function “dictionize” helps in this mapping process by considering the closest match from the available diacritic forms in the dictionary, using a similarity scoring based on character mismatches.

**Accuracy Measurement**

The script includes two accuracy functions that compares the predicted text to the ground truth. The first function “accuracy_wholeWord” calculates the percentage of words that are exactly the same in both the predicted and true texts, while the second and main function “accuracy_char” calculates the correctly guessed potential diacritics characters as requested.

**Additional Functionalities and Handling**

Utility functions like find, without_dia, max_match, and build_dict assist in various string operations, such as finding characters in strings, removing diacritics, choosing the best match for a word, and building a mapping from nondiacritic to diacritic forms.

How to run the “diacritization.py”

The program works in two different modes. The first mode is the training. If you want to train the model first then run the program with the following way:
	“python diacritization.py” this command will train the model, print the both types of accuracy scores on the evaluation dataset and store the model by the default name “diacritization.model” in the same directory where the program is. If you want to store it in the desired name that you want you can use the argument “—model_path” and specify your desired model name.
	
The second mode is the predicting mode, when the program will load the saved model, then will start the interactive session with the user, where the user will enter the text in the STDIN and the program will output the diacritics version of the input to the STDOUT. If you want to close, either terminate with keyboard or hit enter twice which will put an empty character, which will terminate the program. In order to run the program in that way you need the following command:
	“python diacritization.py —predict True —model_path /path/to/model”. The default value for “—model_path” argument is “diacritization.model”.


Accuracy
Accuracy on evaluation data by comparing whole words: 77.99%
MAIN Accuracy on evaluation data by comparing DIA/NODIA characters: 93.13%

How did I get the desired model parameters?

The “diacritization_GridSearch.py”, was the program which I used for the grid searching of the best parameters for this problem. If you run the program “diacritization_GridSearch.py” with command “python diacritization_GridSearch.py” it will output the different model parameters and their accuracies on the development data. It also stores the models with different names on the directory.






