# Feature-Selection-on-Speaker-Recognition-under-Emotional-Conditions

This project tries to find what features are robust for speaker recognition under diffenrent emotional conditions using XGBoost model. 
The main body for this project is [XGBoost_model.ipynb](https://github.com/Cui-ht/Feature-Selection-on-Speaker-Recognition-under-Emotional-Conditions/blob/main/XGBoost_model.ipynb)

## Data
We use the 1440 pure audio files of RAVDESS dataset as training data which includes 8 emotions: calm, happy, sad, angry, fearful, surprise, disgust, and neutral, of which "neutral" serves as a baseline emotion. The dataset also includes both female and male voices and all sentences are spoken in a neutral North American accent.

### Preprocessing
We use eGeMAPs feature set as our pool for selection at first, which are extracted with openSMILE, using the script [smile_egemaps_loop.sh](https://github.com/Cui-ht/Feature-Selection-on-Speaker-Recognition-under-Emotional-Conditions/blob/main/smile_egemaps_loop.sh).
The files names from RAVDESS dataset are decoded with [ravdess_data_preprocess.py](https://github.com/Cui-ht/Feature-Selection-on-Speaker-Recognition-under-Emotional-Conditions/blob/main/ravdess_data_preprocess.py).

## XGBoost Tree
We first use grid search to find the best parameters for the performance of the XGBoost tree, based on which we classify our data to make sure the model works, so that the important features are indeed important.
![confusion_matrix](https://user-images.githubusercontent.com/57549068/205438956-c02950d2-c68b-4036-905c-280c858ee11b.png)
...Which proves that the XGBoost model is reliable.

## Get feature importance
Since we have acquired a reliable model, the important features in the tree model are the ones we want to find. We use various feature importance calculation methods and come up with a final feature set.
