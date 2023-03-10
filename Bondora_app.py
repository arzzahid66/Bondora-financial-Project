import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings
import joblib
warnings.filterwarnings('ignore')

loaded_model =joblib.load("C:/Users/ualal/Random_Regresser", "rb")

def main():
    # giving a title
    st.title('Bondora ROI Estimitor')
    Bids_portfolio_Manager = st.number_input("Bids portfolio Manager")
    Bids_Api= st.number_input("Bids Api")
    Bids_Manual = st.number_input("Bids Manual")
    New_Credit_Customer = st.number_input("New Credit Customer")    
    Age = st.number_input("Age")
    Applied_Amount = st.number_input("Applied Amount")  
    Interest= st.number_input("Interest")  
    Monthly_Payment= st.number_input("Monthly Payment") 
    Income_Total= st.number_input("Income Total") 
    Existing_Liabilities= st.number_input("Existing Liabilities") 
    Refinance_Liabilities= st.number_input("Refinance_Liabilities") 
    Debt_To_Income= st.number_input("Debt To Income") 
    Free_Cash= st.number_input("Free Cash") 
    Restructured= st.number_input("Restructured") 
    Principle_Payment_Made= st.number_input("Principle Payment Made") 
    Interest_And_Penalty_Payments_Made= st.number_input("Interest And Penalty Payments Made") 
    Previous_Early_Repayments_Before_Loan= st.number_input("Previous Early Repayments Before Loan") 
    Verification_Type= st.number_input("Verification Type") 
    Language_Code= st.number_input("Language Code") 
    Gender= st.number_input("Gender") 
    Country= st.number_input("Country") 
    Use_Of_Loan= st.number_input("Use Of Loan") 
    Education=  st.number_input("Education") 
    Marital_Status= st.number_input("Marital Status") 
    Employment_Status= st.number_input("Employment Status")
    Employment_Duration_Current_Employere=st.number_input("Employment Duration Current Employere")
    Occurpation_Area= st.number_input("Occurpation Area")
    Home_Ownership_Type= st.number_input("Home Ownership Type")
    Rating= st.number_input("Rating")
    Credit_Score_Es_MicroL= st.number_input("Credit Score Es MicroL")
    
    
    
    demand_prediction = loaded_model.predict([[Bids_portfolio_Manager,Bids_Api,Bids_Manual,New_Credit_Customer,Age,Applied_Amount,Interest,Monthly_Payment,
                                                   Income_Total,Existing_Liabilities,Refinance_Liabilities,Debt_To_Income,Free_Cash,Restructured,Principle_Payment_Made,Interest_And_Penalty_Payments_Made,
                                                   Interest_And_Penalty_Payments_Made,Previous_Early_Repayments_Before_Loan,Verification_Type,Language_Code,Gender,Country,Use_Of_Loan,Education,Marital_Status,
                                                   Employment_Status,Employment_Duration_Current_Employere,Occurpation_Area,Home_Ownership_Type,Rating,Credit_Score_Es_MicroL,
                                                   ]])

    output = round(demand_prediction[0])
    st.success("Bike Count is {}".format(output))


if __name__ == '__main__':
    main()
    
