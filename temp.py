import numpy as np
import pickle

#loading the saved model
loaded_model=pickle.load(open('C:/machinelearnig/trained_model.sav','rb'))


input_data=[23,1,5,1,7,1,0]

input_data_array=np.asarray(input_data)

input_data_array_reshaped=input_data_array.reshape(1,-1)


prediction=loaded_model.predict(input_data_array_reshaped)

if prediction[0]==0:
    print('not placed')
elif prediction[0]==1:
    print('placed')