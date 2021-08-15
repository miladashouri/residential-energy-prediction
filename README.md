

# Introduction:

This project predicts the energy consumption a building would consume in terms of the given building characteristics.

## Startup:
**The best way to start the application is through creating a conda envireonment, for example:**

```conda create -n residential-building-energy-labeling Python=3.9.6 ```

The dependencies required to run the app are listed in the requirements.txt. simply run the following:

``` pip install -r requirements.txt ```

After installing the required dependencies, run the application by:

```python app.py```


## Using the app:
The fields needed to be filled are listed bellow.

X1: Relative Compactness

X2: Surface Area

X3: Wall Area

X4: Roof Area

X5: Overall Height

X6: Orientation

X7: Glazing Area

X8: Glazing Area Distribution

y1: Heating Load (prediction target)

y2: Cooling Load (prediction target)

The data comes from the open source machine learning repository: [https://archive.ics.uci.edu/ml/datasets/energy+efficiency](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)

An example of such values are:

example= {'Relative_compactness': 0.9,
 'Surface_area': 563.5,
 'Wall_area': 318.5,
 'Roof_area': 122.5,
 'Overall_height': 7.0,
 'Orientation': 4.0,
 'Glazing_area': 0.25,
 'Glazing_area_distribution': 3.0}

 with corresponding energy indices:
 
 Heating Load = {"Heating_Load": 32.68}
 
 Cooling Load= {"Cooling_Load": 32.83}

 You can try them using the trained model by filling the required inputs.

Here are the snapshots of the home page and prediction page:

![alt text](https://github.com/miladashouri/residential-energy-prediction/blob/master/home_page.PNG "Logo Title Text 1")

![alt text](https://github.com/miladashouri/residential-energy-prediction/blob/master/prediction_page.PNG "Logo Title Text 2")

## Deployment:
Docker is recommended to deploy the flask app (development environment). A dockerfile and docker-compose.yml has been provided. Simply, create a server and install docker. Then run the following where the docker-compose.yml file exists. 

``` sudo docker-compose up --build ``` 

The application is available on the server (port=80).

## Training:

```python app.py```

The train.py file fetches the data (from data folder) and runs machine learning models (cuurently random forest) and saves the data and parameters in pickle file and joblib format. The models are placed in the models directory.  
