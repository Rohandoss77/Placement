# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:12:28 2024

@author: DGVC
"""

import numpy as np
import streamlit as st
import pickle


loaded_model=pickle.load(open('trained_model.sav','rb'))


def placementprediction(input_data):
    input_data=[23,1,5,1,7,1,0]
    
    input_data_array=np.asarray(input_data)
    
    input_data_array_reshaped=input_data_array.reshape(1,-1)
    
    
    prediction=loaded_model.predict(input_data_array_reshaped)
    
    if prediction[0]==0:
        print('not placed')
    elif prediction[0]==1:
        print('placed')
        
    
def main():
    st.title('placement prediction web app')
    
    Age=st.text_input('Age')
    Gender=st.text_input('Gender')
    Stream=st.text_input('Stream')
    Internships=st.text_input('Internships')
    CGPA =st.text_input('CGPA')
    Hostel=st.text_input('Hostel')
    HistoryOfBacklogs=st.text_input('HistoryOfBacklogs')
    PlacedOrNot=st.text_input('PlacedOrNot')
    
    #prediction
    predictions=''
    
    if st.button('placement prediction'):
        predictions=placementprediction([Age,Gender,Stream,Internships,CGPA,Hostel,HistoryOfBacklogs,PlacedOrNot])
        
    st.success(predictions)
    
if __name__ == '__main__':
    main()
