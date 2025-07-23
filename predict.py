# buat program predict.py untuk memprediksi harga mobil menggunakan model yang sudah dilatih
# library yang digunakan pickle, numpy, pandas, xgboost, sklearn dan streamlit

# Deploy Churn Predictor

# ======================================================
import pandas as pd
import numpy as np

from xgboost.sklearn import XGBClassifier
import streamlit as st
import pickle


# ======================================================

# judul aplikasi
st.header('Saudi Arabia Used Cars Price Prediction')

# tambahkan sidebar
st.sidebar.header('Input Features')

# buat fungsi untuk mengambil input dari pengguna
def user_input_features():
    Region = st.sidebar.selectbox(label='Region', options = ['Abha', 'Al-Ahsa', 'Al-Baha', 'Al-Jouf', 'Al-Medina', 'Al-Namas', 'Arar','Aseer', 'Besha', 
                                                             'Dammam', 'Hafar Al-Batin', 'Hail', 'Jazan', 'Jeddah', 'Jubail', 'Khobar', 'Makkah', 'Najran', 
                                                             'Qassim', 'Qurayyat', 'Riyadh', 'Sabya', 'Sakaka', 'Tabouk', 'Taef', 'Wadi Dawasir', 'Yanbu'])
    Make = st.sidebar.selectbox(label='Make', options = ['Toyota', 'GMC', 'Land Rover', 'Kia', 'Mazda', 'Porsche',
       'Hyundai', 'Lexus', 'Chrysler', 'Chevrolet', 'Nissan',
       'Mitsubishi', 'Ford', 'MG', 'Mercedes', 'Jeep', 'BMW', 'Audi',
       'Lincoln', 'Cadillac', 'Genesis', 'Renault', 'Honda', 'Suzuki',
       'Zhengzhou', 'Dodge', 'HAVAL', 'INFINITI', 'Isuzu', 'Changan',
       'Aston Martin', 'Mercury', 'Great Wall', 'Other', 'Rolls-Royce',
       'MINI', 'Volkswagen', 'BYD', 'Geely', 'Victory Auto', 'Classic',
       'Jaguar', 'Daihatsu', 'Maserati', 'Hummer', 'GAC', 'Lifan',
       'Bentley', 'Chery', 'Peugeot', 'Foton', 'Škoda', 'Fiat', 'Iveco',
       'SsangYong', 'FAW', 'Tata', 'Ferrari'])
    Type = st.sidebar.selectbox(label='Type', options = ['Corolla', 'Yukon', 'Range Rover', 'Optima', 'FJ', 'CX3',
       'Cayenne S', 'Sonata', 'Avalon', 'LS', 'C300', 'Land Cruiser',
       'Hilux', 'Tucson', 'Caprice', 'Sunny', 'Pajero', 'Azera', 'Focus',
       '5', 'Spark', 'Camry', 'Pathfinder', 'Accent', 'ML', 'Tahoe',
       'Yaris', 'Suburban', 'A', 'Altima', 'Traverse', 'Expedition',
       'Senta fe', 'Liberty', '3', 'X', 'Elantra', 'Land Cruiser Pickup',
       'VTC', 'Malibu', 'The 5', 'A8', 'Patrol', 'Grand Cherokee', 'SL',
       'Previa', 'SEL', 'Aveo', 'MKZ', 'Victoria', 'Datsun', 'Flex',
       'GLC', 'ES', 'Edge', '6', 'Escalade', 'Innova', 'Navara', 'H1',
       'G80', 'Carnival', 'Symbol', 'Camaro', 'Accord', 'Avanza',
       'Land Cruiser 70', 'Taurus', 'C5700', 'Impala', 'Optra', 'S',
       'Other', 'Cerato', 'Furniture', 'Murano', 'Explorer', 'LX',
       'Pick up', 'Charger', 'H6', 'BT-50', 'Hiace', 'Ranger', 'Fusion',
       'Rav4', 'Ciocca', 'CX9', 'Kona', 'Sentra', 'Sierra', 'Durango',
       'CT-S', 'Sylvian Bus', 'Navigator', 'Opirus', 'Marquis', 'The 7',
       'FX', 'Creta', 'D-MAX', 'CS35', 'The 3', 'Dyna', 'GLE', 'Sedona',
       'Prestige', 'CLA', 'Lumina', 'Vanquish', 'Sorento', 'Safrane',
       'Cores', 'Cruze', 'Prado', 'Cadenza', "D'max", 'Silverado', 'Rio',
       'Maxima', 'X-Trail', 'RX', 'Cressida', 'C', 'Seven', 'Cherokee',
       'Grand Marquis', 'H2', 'QX', 'Blazer', 'Wingle', 'Panamera',
       'Rush', 'The M', 'Genesis', 'E', 'K5', 'CS95', 'Cayenne Turbo S',
       'Civic', 'Echo Sport', 'Challenger', 'CL', 'Wrangler', 'A6',
       'Dokker', 'CX5', 'Mohave', 'Ghost', 'Copper', 'Veloster', 'G',
       'Jetta', 'IS', 'Thunderbird', 'Fluence', 'V7', 'Vego', 'Aurion',
       'Q', 'F3', 'UX', 'Beetle', 'F150', 'Acadia', 'EC7', 'Lancer',
       'Capture', 'Van R', 'Mustang', 'CS35 Plus', 'DB9', 'APV',
       'Kaptiva', 'Viano', 'Safari', 'Cadillac', 'CLS', 'Duster',
       'Platinum', 'Carenz', 'Emgrand', 'Z', 'Coupe S', 'Odyssey',
       'Terrain', 'Juke', 'Sportage', 'C200', 'Attrage', 'GS', 'X-Terra',
       'Azkarra', 'XF', 'Picanto', 'Armada', 'CT5', 'KICKS', 'Gran Max',
       'Cayman', 'Levante', 'Montero', '300', 'POS24', 'A3', 'Touareg',
       'Passat', 'Delta', 'H3', 'RX5', 'GS3', 'Coupe', 'New Yorker',
       'Cayenne Turbo', 'Colorado', 'Trailblazer', 'Vitara', 'Nativa',
       'Van', 'LF X60', 'Koleos', 'Defender', 'Abeka', 'H100',
       'Flying Spur', 'Pilot', 'L200', 'A7', 'Quattroporte', 'Bora',
       'Compass', 'Bus Urvan', 'Macan', 'Corolla Cross', 'GL', 'City',
       'DTS', 'Ertiga', 'Envoy', 'CT6', 'Fleetwood', 'Tiggo', 'GX', 'Q5',
       'A4', 'XJ', 'Echo', 'HS', 'Avalanche', 'MKX', 'Seltos', 'SRX',
       'RX8', 'SLK', '301', 'EC8', '3008', 'Suvana', 'Prius', 'Cayenne',
       'Eado', 'The 6', 'Royal', 'NX', 'Soul', 'CS75', 'H9', 'F-Pace',
       'Coolray', 'Maybach', 'CS85', 'Jimny', 'GC7', '360', 'A5', 'S300',
       'Superb', 'Ram', 'The 4', 'Grand Vitara', '500', 'Logan', '5008',
       'Tiguan', 'Golf', 'S5', '911', 'Boxer', 'Camargue', 'M', 'Daily',
       'Nitro', 'CRV', 'Mini Van', 'Pegas', 'L300', 'Coaster',
       'Discovery', 'Montero2', 'Bentayga', 'Z370', 'Bus County',
       'Stinger', 'SRT', 'CT4', 'F Type', 'CC', 'Koranado', 'ASX',
       'Carens', 'Crown', 'ِACTIS V80', 'XT5', 'Tuscani', '4Runner',
       'ATS', 'HRV', 'X7', 'Outlander', 'X40', 'Q7', 'ZS', 'G70',
       'Megane', 'Nexon', 'Power', 'B50', 'Town Car', '2', 'i40', 'RC',
       'Doblo', 'Bronco', 'Dzire', 'Avante', 'Z350', 'CX7', 'Countryman',
       'GTB 599 Fiorano', 'Prestige Plus', 'Terios', 'MKS', 'Milan',
       'Centennial', 'Dakota', 'Savana', 'S8'])
    
    Origin = st.sidebar.selectbox(label='Origin', options = ['Saudi', 'Gulf Arabic', 'Other'])
    Gear_Type = st.sidebar.selectbox(label='Gear Type', options = ['Automatic', 'Manual'])
    Options = st.sidebar.selectbox(label='Options', options = ['Standard', 'Semi Full', 'Full'])
    Year = st.sidebar.number_input(label='Year', min_value=1990, max_value=2024, value=2015)
    Mileage = st.sidebar.number_input(label='Mileage', min_value=0, max_value=1000000, value=50000)
    Engine_Size = st.sidebar.number_input(label='Engine Size', min_value=0.0, max_value=10.0, value=2.0, step=0.1)
 

    df = pd.DataFrame({
        'Region': [Region],
        'Make': [Make],
        'Type': [Type],
        'Origin': [Origin],
        'Gear_Type': [Gear_Type],
        'Options': [Options],
        'Year': [Year],
        'Mileage': [Mileage],
        'Engine_Size': [Engine_Size]
    })

    return df

df_predict = user_input_features()
df_predict.index = ['Value']
# predict harga mobil menggunakan model yang sudah dilatih
loaded_model = pickle.load(open('xgboost_model_final.sav', 'rb'))
prediction = loaded_model.predict(df_predict)

# tampilkan hasil prediksi
column1, column2 = st.columns(2)

# Column 1: tampilkan input fitur
with column1:
    st.subheader('Input Features')
    st.write(df_predict.T)
# Column 2: tampilkan hasil prediksi
with column2:
    st.subheader('Predicted Price')
    st.subheader(f'{prediction[0]:,.2f}')  # Format harga dengan tanda dolar dan dua desimal

    # tampilkan nilai evaluasi model
    st.subheader('Model Evaluation')
    st.write('MSE:', 756917504)
    st.write('RMSE:', 27512.1)
    st.write('MAE:', 16072.3)
   
   # tampilkann range harga prediksi
    st.subheader('Price Range')
    st.write(f'Predicted Price Range: {prediction[0] - 16072.3:,.2f} - {prediction[0] + 27512.1:,.2f}')  # Format harga dengan tanda dolar dan dua desimal
    
    