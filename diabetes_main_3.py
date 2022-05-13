# Code for 'diabetes_main.py' file.

# Import the necessary modules design the Decision Tree classifier
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import GridSearchCV  
from sklearn import tree
from sklearn import metrics

# Configure your home page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(page_title = 'Early Diabetes Prediction Web App',
                    page_icon = 'random',
                    layout = 'wide',
                    initial_sidebar_state = 'auto'
                    )

# Loading the dataset.
@st.cache()
def load_data():
    # Load the Diabetes dataset into DataFrame.

    df = pd.read_csv('b510b80d-2fd6-4c08-bfdf-2a24f733551d.csv')
    df.head()

    # Rename the column names in the DataFrame.
    df.rename(columns = {"BloodPressure": "Blood_Pressure",}, inplace = True)
    df.rename(columns = {"SkinThickness": "Skin_Thickness",}, inplace = True)
    df.rename(columns = {"DiabetesPedigreeFunction": "Pedigree_Function",}, inplace = True)

    df.head() 

    return df

diabetes_df = load_data()






@st.cache()
def d_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age):
    # Split the train and test dataset. 
    feature_columns = list(diabetes_df.columns)

    # Remove the 'Pregnancies', Skin_Thickness' columns and the 'target' column from the feature columns
    feature_columns.remove('Skin_Thickness')
    feature_columns.remove('Pregnancies')
    feature_columns.remove('Outcome')

    X = diabetes_df[feature_columns]
    y = diabetes_df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    dtree_clf.fit(X_train, y_train) 
    y_train_pred = dtree_clf.predict(X_train)
    y_test_pred = dtree_clf.predict(X_test)
    # Predict diabetes using the 'predict()' function.
    prediction = dtree_clf.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(metrics.accuracy_score(y_train, y_train_pred) * 100, 3)

    return prediction, score

st.markdown("<p style='color:red;font-size:25px'>This app uses <b>Decision Tree Classifier</b> for the Early Prediction of Diabetes.", unsafe_allow_html = True) 
st.subheader("Select Values:")
    
   # Create the user defined 'app()' function.

    
    # Create six sliders with the respective minimum and maximum values of features. 
    # store them in variables 'glucose', 'bp', 'insulin, 'bmi', 'pedigree' and 'age'
    # Write your code here:
glucose = st.slider("glucose", 0.0, 10.0)
bp = st.slider("bp", 0.0, 10.0)
bmi = st.slider("bmi", 0.0, 10.0)
insulin=st.slider("insulin",0.0,10.0)
pedigree= st.slider("pedigree", 0.0, 10.0)
age = st.slider("age", 0.0, 10.0)


    

    # Add a single select drop down menu with label 'Select the Classifier'
if st.button("Predict"):

	prediction, score = d_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age)
	if prediction ==1:
		st.info("The person either has diabetes or prone to get diabetes")
	else:
		st.info("the person is free from diabetes")
	st.write("The accuracy score of this model is", score, "%")	
		

    

	 
     	 

               
  	 

     
     
     



           
         
            


  

    
        



         
            