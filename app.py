import streamlit as st
import pandas as pd
import numpy as np
import dataikuapi
import altair as alt
from PIL import Image

st.title('Lending Webapp')

image = Image.open('./Images/dataiku.jpg')
st.sidebar.image(image)
st.sidebar.text('simple app to call a model built \n and hosted by Dataiku')

st.write('Please choose variable inputs: ')
#Input widget functions for HMEQ
def user_input_features():
    LOAN = st.slider('Loan Amount', 1000, 90000, 20000)
    FICO = st.slider('Fico Score', 300, 850, 500)
    TERM = st.selectbox('Term Length', 
                    ('36 months', '60 months'))
    HOME = st.selectbox('Home Ownership', 
                    ('MORTGAGE', 'RENT', 'OWN'))
    DTI = st.slider('Debt to Income ratio', 0.0, 100.0, 25.0, .5)
    INCOME = st.text_input('Annual Income', placeholder=50000)

    data = {
        "id": 164193225,
        "dti": DTI,
        "delinq_2yrs": 0,
        "earliest_cr_line": "Sep-1997",
        "earliest_cr_line_parsed": "1997-09-01T00:00:00.000Z",
        "age_credit_history": 295,
        "earliest_cr_line_parsed_year": 1997,
        "earliest_cr_line_parsed_month": 9,
        "earliest_cr_line_parsed_day": 1,
        "avg_fico": FICO,
        "inq_last_6mths": 1,
        "mths_since_last_delinq": 27,
        "open_acc": 18,
        "pub_rec": 0,
        "acc_now_delinq": 0,
        "all_util": 90,
        "inq_last_12m": 4,
        "chargeoff_within_12_mths": 0,
        "num_accts_ever_120_pd": 2,
        "emp_title": "Rn",
        "emp_length": "4-9 years",
        "home_ownership": HOME,
        "annual_inc": INCOME,
        "verification_status": True,
        "addr_state": "CA",
        "mort_acc": 0,
        "loan_amnt": LOAN,
        "term": TERM,
        "int_rate": "12.40%",
        "purpose": "credit_card",
        "title": "Credit card refinancing",
        "Default": 0
        }
    return data

record_to_predict = user_input_features()

if st.button('Score'):
    try:
        
        client = dataikuapi.APINodeClient("http://localhost:11800", "lending_demo_streamlit")
        prediction = client.predict_record("streamlitmodel", record_to_predict)
        explanations = prediction["result"]["explanations"]
        # print(prediction)
        shap = pd.DataFrame.from_dict(explanations, orient='index')
        shap.reset_index(inplace=True)
        shap = shap.rename(columns = {
            'index': 'labels',
            0: 'data'
        })
        # st.write(prediction["result"])
        st.write('The predicted value for {} is: {}'.format('Default',prediction['result']['prediction']))
        
        st.write('The most influential variables contributing are:')

        c = alt.Chart(shap).mark_bar().encode(
        x=alt.X('labels', title='Top Predictors'),
        y=alt.Y('data', title='Estimated Impact (Shapley)'),
        color=alt.Color('labels'), tooltip=['labels', 'data']
        )
        st.altair_chart(c, use_container_width=True)
    except: 
        st.write("Having trouble connecting to the prediction endpoint")