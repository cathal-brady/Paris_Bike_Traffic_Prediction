# Cathal & Nadiy Bike Prediction
Welcome to our repo for predicting the log hourly traffic at each counter from the 10-09-2021 to the 18-10-2021.

The final Python script can be found in the code folder (Code/final_model_12_10.py). The script was last run on Kaggle so if are running it locally you will need to be change the file paths from where data is collected. These paths have been moved to the top of the script for convenience. 

The script has passed flake8 and pydocstyle tests, as well as having significant docstrings for each function/class

## Some further Info:
All data used is in the **datasets** folder. This includes, weather data from both Paris and Orly Airport, Covid Data, Covid Lockdown Data, and Multimodal Traffic Data. Links to these data sources which were not given in the project brief can be found in the text file (data_source_links.txt), in the datasets folder.

The **code** folder contains our final Python script (final_model_12_10.py) which is the final result of much iteration. Data was cleaned and investigated in Python notebooks which are also in this folder and can be a useful source of information regarding how the data was merged and cleaned etc. but these are not needed to fit the model. 

There is also a folder called **Trained_Models** this is used to dump joblib files for each site_ID into and these are then recollected later for final predictions. If running the code locally you will need to have this folder in place. 

If you have any questions - dont hesitate to reach out to us @ cathal.brady@polytechnique.edu 
