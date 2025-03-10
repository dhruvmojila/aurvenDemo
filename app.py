import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

def getLLMOutput(userInput, output):
    system_prompt = "You are a helpful airline agent."
    augmented_query= f"""
    I provided you user input and output. Input include user choice for source and destination, class, stop type, airline. and output include as per their input on which day of a week they get checpest ticket. 

    ==============================================
    user input: 
    {userInput}

    model:
    {output}

    Your task is to simply anlyse input and outpu and give user a good reposne. in natual language. is must be one liner.

    you can say something like: Hello [greetings], as per you source and desination with requiremnt of class and stop type you might get chepest tickets on this day [model output]. 

    Make your response such as professional airline agent and attractive. no pre or post amble.
    """
    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content

def load_model():
    print("loacing model")
    with open("random_forest_model.pkl", "rb") as model_file:
        print(" model loaded")
        return pickle.load(model_file)
model = load_model()
# Dropdown options
source_options = ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Kolkata', 'Chennai', 'Ahmedabad']
destination_options = ['Mumbai', 'Bangalore', 'Hyderabad', 'Kolkata', 'Chennai', 'Ahmedabad', 'Delhi']
airlineMap = {'SpiceJet': 0, 'Indigo': 1, 'GO FIRST': 2, 'Air India': 3, 'AirAsia': 4, 'Vistara': 5, 'AkasaAir': 6, 'AllianceAir': 7, 'StarAir': 8}
stop_mapping = {"non-stop": 0, "1-stop": 1, "2+-stop": 2}
class_mapping = {"Economy": 0, "Premium Economy": 1, "Business": 2, "First": 3}

def preprocess_input(user_input):
    user_input_df = pd.DataFrame([user_input])
    user_input_df['class'] = user_input_df['class'].map(class_mapping)
    user_input_df['total_stops'] = user_input_df['total_stops'].map(stop_mapping)
    user_input_df['airline'] = user_input_df['airline'].map(airlineMap)
    user_input_df = pd.get_dummies(user_input_df, columns=['source', 'destination'])
    
    model_columns = ['source_Bangalore', 'source_Chennai', 'source_Delhi', 'source_Hyderabad', 'source_Kolkata', 'source_Mumbai',
                     'destination_Bangalore', 'destination_Chennai', 'destination_Delhi', 'destination_Hyderabad', 'destination_Kolkata', 'destination_Mumbai']
    for col in model_columns:
        if col not in user_input_df.columns:
            user_input_df[col] = 0
    
    return user_input_df

def predict_with_model(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    dayMap = {0: 'Wednesday', 1: 'Saturday', 2: 'Friday', 3: 'Sunday', 4: 'Tuesday', 5: 'Monday', 6: 'Thursday'}
    output = model.predict(input_array)[0]
    return dayMap[output]

# Streamlit App
st.title("Flight Prediction App")

source = st.selectbox("Select Source", source_options)
destination = st.selectbox("Select Destination", destination_options)
airline = st.selectbox("Select Airline", list(airlineMap.keys()))
total_stops = st.selectbox("Select Number of Stops", list(stop_mapping.keys()))
flight_class = st.selectbox("Select Class", list(class_mapping.keys()))

if st.button("Predict Flight Day"):
    user_input = {
        "source": source,
        "destination": destination,
        "class": flight_class,
        "total_stops": total_stops,
        "airline": airline
    }
    processed_input = preprocess_input(user_input)
    prediction = predict_with_model(processed_input)
    llmans = getLLMOutput(user_input,prediction )
    st.success(f"{llmans}")
