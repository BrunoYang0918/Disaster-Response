# Disaster Response
This purpose to build this natural language model is to increase the speed of classify text message to the right category.

## Initialization
The below pacakges are required for running these script.
sys, sklearn, re, ntlk, pandas, numpy, sqlalchemy

## File Description
1. process_data.py: 
  A data clean function, which takes in raw data and output the combined datasets used in the modeling process. 
  A typical to call this funciton is like "python process_data.py disaster_messages.csv disaster_categories.csv"
  
2. train_classifier.py:
  A modeling function, which use the cleaned dataset to create a text classification model.
  A typical to call this funciton is like "python train_classifier.py DisasterResponse.db classifier.pkl"
