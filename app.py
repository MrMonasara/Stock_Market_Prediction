import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin.stock_info import get_data
from streamlit_option_menu import option_menu

selected = option_menu(None,
    options=["Home", "Stock Predictor", "About"],
    icons=["house", "book", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

def get_stock_data(ticker):
    df = get_data(ticker, start_date = None, end_date = None, index_as_date = False, interval ='1d')
    return df

##------------------------------- HOME PAGE -------------------------------
if selected == "Home":
    st.title("Welcome to the Stock Predictor!!")
    st.write("Now have a better glance of market with this ML powered stock price predictor & invest better.")

##------------------------------- PREDICTION PAGE -------------------------------
elif selected == "Stock Predictor":
    st.title("Predictor")
    ticker = st.selectbox("Pick any stock or index to predict:" ,
        ("APOLLOHOSP.NS","TATACONSUM.NS","TATASTEEL.NS","RELIANCE.NS","LT.NS","BAJAJ-AUTO.NS","WIPRO.NS","BAJAJFINSV.NS","KOTAKBANK.NS",
        "ULTRACEMCO.NS","BRITANNIA.NS","TITAN.NS","INDUSINDBK.NS","ICICIBANK.NS","ONGC.NS","NTPC.NS","ITC.NS","BAJFINANCE.NS","NESTLEIND.NS",
        "TECHM.NS","HDFCLIFE.NS","HINDALCO.NS","BHARTIARTL.NS","CIPLA.NS","TCS.NS","ADANIENT.NS","HEROMOTOCO.NS","MARUTI.NS","COALINDIA.NS",
        "BPCL.NS","HCLTECH.NS","ADANIPORTS.NS","DRREDDY.NS","EICHERMOT.NS","ASIANPAINT.NS","GRASIM.NS","JSWSTEEL.NS","DIVISLAB.NS","TATACONSUM.NS",
        "SBIN.NS","HDFCBANK.NS","HDFC.NS","WIPRO.NS","UPL.NS","POWERGRID.NS","TATAPOWER.NS","TATAMOTORS.NS","SUNPHARMA.NS","HINDUNILVR.NS",
        "SBILIFE.NS","INFY.NS","AXISBANK.NS"))
    
    from_date = st.date_input("From Date:",datetime.date(2019, 7, 6))
    to_date = st.date_input("To Date:",datetime.date(2019, 7, 6))
    
    st.button("Predict", get_stock_data)
    st.write('You selected:', ticker)

    df = get_stock_data(ticker)
    st.dataframe(df)
    st.line_chart(data=df, x='date', y='close', use_container_width=True)



    ## ------------------------------- PREDICTION LOGIC -------------------------------
    X = df[['open','high','low']]
    y = df['close'].values.reshape(-1,1)

    df = df.replace(0, pd.np.nan).dropna(axis = 0, how ='any')
    df.drop(['volume'], axis=1)
    
    #Splitting our dataset to Training and Testing dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Fitting Linear Regression to the training set
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    #predicting the Test set result
    y_pred = reg.predict(X_test)
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values

    pred = []
    for i in range(0,5147):
        open = o[i]
        high = h[i]
        low = l[i]
        output = reg.predict([[open,high,low]])
        pred.append(output)

    pred1 = np.concatenate(pred)
    predicted = pred1.flatten().tolist()


##------------------------------- ABOUT US PAGE -------------------------------
elif selected == "About":
    st.title("About Us")
    st.write("This UI is made to predict the close price based upon the previous data.")