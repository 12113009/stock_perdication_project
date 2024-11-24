import pandas_datareader as data
import numpy as np 
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt 
from keras.models import load_model 
import streamlit  as st 

stock=st.title('Stock Tends Predication')
stock=st.text_input('Enter the stock Ticker',"GOOG")

from datetime import datetime
end=datetime.now()
start=datetime(end.year-20,end.month,end.day) 

df=yf.download(stock,start,end)
#model1=load_model("")
st.subheader('Stock Data')
st.write(df)

st.subheader('Closing Price vs Time chart')
fig =plt.figure(figsize=(15,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 =df.Close.rolling(100).mean()
fig =plt.figure(figsize=(15,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 =df.Close.rolling(100).mean()
ma200 =df.Close.rolling(200).mean()
fig =plt.figure(figsize=(15,6))
plt.plot(ma100,'g')
plt.plot(ma200,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)

# Splitting Data into Testing and Training.

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler(feature_range=(0,1))

data_training_arr=scaler.fit_transform(data_training)

#Load my model 
model=load_model('sekhar_model.keras')
#Testing Part
past_100_days_data=data_training.tail(100)

final_df=pd.concat([past_100_days_data,data_testing],ignore_index=True)

input_data =scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])
  # Convert to NumPy arrays after the loop
x_test, y_test = np.array(x_test), np.array(y_test)

my_y_perdications = model.predict(x_test)

scaler=scaler.scale_

scale_factor = 1/scaler[0]
my_y_perdications = my_y_perdications * scale_factor
y_test=y_test * scale_factor

#Final Graph
st.subheader('Predictions vs Original Price')
fig2=plt.figure(figsize=(15, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(my_y_perdications, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



