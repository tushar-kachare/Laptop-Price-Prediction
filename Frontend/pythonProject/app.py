import numpy as np
import streamlit as st
import sklearn
import pandas as pd
import pickle
st.title("Laptop Price Prediction")
with open('Frontend/pythonProject/data.pkl', 'rb') as f:
    data = pickle.load(f)
with open('Frontend/pythonProject/pipe.pkl','rb') as f:
    model = pickle.load(f)

#Select Company Name
Company = st.selectbox("Comapny Name",data['Company'].unique())

#Select Brand Name
Brand = st.selectbox("Brand Name",data['TypeName'].unique())

#Ram
Ram = st.selectbox("Ram in (GB)",[2,4,6,8,12,16,24,32,64])


#Operating System
Oprating_System = st.selectbox("Select Operating System",data['OpSys'].unique())

#Weight
weight = st.number_input("Enter Weight of Laptop")

#Touchscreen
touchscreen = st.selectbox("Touchscreen",['No','Yes'])

#Resolution
Resolution = st.selectbox("Select Specified Resolution",['2560x1600',
                    '1440x900','1920x1080','2880x1800','1366x768','2304x1440','3200x1800',
                    '1920x1200','2256x1504','3840x2160','2160x1440','2560x1440','1600x900',
                    '2736x1824','2400x1600'])

#IPS
ips = st.selectbox("IPS Display",['No','Yes'])

#HD
Full_Hd = st.selectbox("Full Hd Display",['No','Yes'])

#Ghz
Ghz = st.number_input("Cpu Ghz eg. 3.0, 2.75")

#Cpu Brand Name
Cpu_Brand_Name = st.selectbox("Cpu Brand Name", data['Cpu Brand'].unique())

#ssd
SSD_GB = st.number_input("SSD in GB")

#hdd
HDD_GB = st.number_input("HDD in GB")

#Gpu
Gpu_Brand = st.selectbox("Select Gpu Brand",data['Gpu Brand'].unique())

if st.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    if Full_Hd == 'Yes':
        Full_Hd = 1
    else:
        Full_Hd = 0
    res_split = Resolution.split('x')
    x_res = int(res_split[0])
    y_res = int(res_split[1])
    Resolution = x_res * y_res
    ip = np.array([Company,Brand,Ram,Oprating_System,weight,touchscreen
                      ,Resolution,ips,Full_Hd,Ghz,Cpu_Brand_Name,SSD_GB,HDD_GB,Gpu_Brand])
    ip = ip.reshape(1,14)
    ans = model.predict(ip)
    st.title("Predicted Price of Laptop in Rs : " + str(np.round(np.exp(ans),2)))
