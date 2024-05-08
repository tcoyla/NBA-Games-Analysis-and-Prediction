# NBA-Games-Analysis-and-Prediction
NBA games analysis through an interactive dashboard developed with Dash and Plotly. Machine learning prediction model to determine the number of wins for each team in a regular season.

This project has been done in Python language, it contains : 

- A data folder with 4 csv files that contain all data and a 'logos' folder containing all the images used in the dashboard. 
- A notebook (predictions_model.ipynb) containing all the explanation of the project and the creation and comparison of each machine learning models.
- A PFE.pdf file  contains a presentation of the project.
- A logos_image_analysis.py file which is a little script to extract the logo of each team from the original image containing all the logos at the same place.
- A dash_prediction.py file containing a code to build the best prediction model for each season.
- A main.py file containing the interactive dashboard.

To run the project you have to create a Python environment with version==3.8 (you can use this command line : _conda create -n myEnv python==3.8_) and install all the packages in it with the followed command line : _pip install -r requirements.txt_

To access the interactive dashboard : run the main.py file and enter the following address http://localhost:8050/  in the web navigator of your choice.
