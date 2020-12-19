from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_model')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.jpg')
    image_hospital = Image.open('logo2.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict Airline Passenger Satisfaction')
    
    st.sidebar.image(image_hospital)

    st.title("Airline Passenger Satisfaction Prediction App")

    if add_selectbox == 'Online':
        id = st.number_input('ID', min_value=1, max_value=10000, value=122)
        Gender = st.selectbox('Gender', ['Male','Female'])
        CustomerType = st.selectbox('Customer Type', ['Loyal Customer','disloyal Customer'])
        Age = st.number_input('Age', min_value=1, max_value=100, value=25)
        TypeofTravel = st.selectbox('Type of Travel', ['Business travel','Personal Travel'])
        Class = st.selectbox('Class', ['Eco','Business','Eco Plus'])
        FlightDistance = st.number_input('Flight Distance', min_value=1, max_value=100000, value=25)
        Inflightwifiservice = st.selectbox('Inflight wifi service', [0,1,2,3,4,5])
        DepartureArrivaltimeconvenient = st.selectbox('Departure/Arrival time convenient', [0,1,2,3,4,5])
        EaseofOnlinebooking = st.selectbox('Ease of Online booking', [0,1,2,3,4,5])
        Gatelocation = st.selectbox('Gate location', [0,1,2,3,4,5])
        Foodanddrink = st.selectbox('Food and drink', [0,1,2,3,4,5])
        Onlineboarding = st.selectbox('Online boarding', [0,1,2,3,4,5])
        Seatcomfort = st.selectbox('Seat comfort', [0,1,2,3,4,5])
        Inflightentertainment = st.selectbox('Inflight entertainment', [0,1,2,3,4,5])
        Onboardservice = st.selectbox('On-board service', [0,1,2,3,4,5])
        Legroomservice = st.selectbox('Leg room service', [0,1,2,3,4,5])
        Baggagehandling = st.selectbox('Baggage handling', [0,1,2,3,4,5])
        Checkinservice = st.selectbox('Checkin service', [0,1,2,3,4,5])
        Inflightservice = st.selectbox('Inflight service', [0,1,2,3,4,5])
        Cleanliness = st.selectbox('Cleanliness', [0,1,2,3,4,5])
        DepartureDelayinMinutes = st.number_input('Departure Delay in Minutes', min_value=1, max_value=100000, value=25)
        ArrivalDelayinMinutes = st.number_input('Arrival Delay in Minutes', min_value=1, max_value=100000, value=25)

        
        output=""

        input_dict = {'id':id,'Gender' : Gender, 'CustomerType' : CustomerType, 'Age' : Age, 'TypeofTravel' : TypeofTravel,'Class' : Class,'FlightDistance':FlightDistance,'Inflightwifiservice':Inflightwifiservice,'DepartureArrivaltimeconvenient':DepartureArrivaltimeconvenient ,'EaseofOnlinebooking':EaseofOnlinebooking ,'Gatelocation':Gatelocation ,'Foodanddrink':Foodanddrink,'Onlineboarding':Onlineboarding,'Seatcomfort':Seatcomfort,'Inflightentertainment':Inflightentertainment,'Onboardservice':Onboardservice,'Legroomservice':Legroomservice,'Baggagehandling':Baggagehandling,'Checkinservice':Checkinservice,'Inflightservice':Inflightservice,'Cleanliness':Cleanliness,'DepartureDelayinMinutes':DepartureDelayinMinutes,'ArrivalDelayinMinutes':ArrivalDelayinMinutes}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()