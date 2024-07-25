import pandas as pd
import streamlit as st
import requests
import numpy as np
import pickle
from datetime import datetime, timedelta

# Load the trained ARIMA model
try:
    with open('model/arima_model.pkl', 'rb') as file:
        model_ARIMA = pickle.load(file)
    st.success('Model ARIMA berhasil dimuat')
except FileNotFoundError:
    st.error('File model tidak ditemukan. Pastikan path file benar.')
    st.stop()

st.title('Data Historis Saham PT. Telkom')

# Load historical data
try:
    df_historical = pd.read_csv('D:\Proyek Data Mining\stock_prediction\data\Data Historis TLKM.csv')  # Pastikan path file benar
    st.write("Data historis:")
    st.write(df_historical.head())  # Tampilkan beberapa baris pertama untuk verifikasi
    
    if 'Tanggal' not in df_historical.columns:
        raise KeyError("Kolom 'Tanggal' tidak ditemukan dalam data historis.")
    
    df_historical['Tanggal'] = pd.to_datetime(df_historical['Tanggal'])
    df_historical.set_index('Tanggal', inplace=True)
    st.success('Data historis berhasil dimuat')
except FileNotFoundError:
    st.error('File data historis tidak ditemukan. Pastikan path file benar.')
    st.stop()
except KeyError as e:
    st.error(str(e))
    st.stop()

start_date = st.date_input('Start Date', value=datetime(2021, 1, 1))
end_date = st.date_input('End Date', value=datetime.today())

def predict(start_date, end_date):
    try:
        # Generate date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        num_dates = len(date_range)

        # Use the ARIMA model to forecast the stock prices
        forecast_result = model_ARIMA.get_forecast(steps=num_dates)
        forecast = forecast_result.predicted_mean
        
        # Ensure forecast list length matches date range length
        if len(forecast) != num_dates:
            raise ValueError("Length of forecast does not match length of date range.")

        # Prepare the results
        results = {
            'Tanggal': date_range.strftime('%Y-%m-%d').tolist(),
            'predictions': forecast.tolist()
        }
        
        return results
    except Exception as e:
        return {'error': str(e)}

if st.button('Prediksi'):
    if start_date and end_date:
        st.write(f'Start Date: {start_date}')
        st.write(f'End Date: {end_date}')
        results = predict(start_date, end_date)
        if 'error' in results:
            st.write('Terjadi kesalahan:', results['error'])
        else:
            # Create DataFrame from results
            df_predictions = pd.DataFrame(results)
            df_predictions['Tanggal'] = pd.to_datetime(df_predictions['Tanggal'])  # Convert date strings to datetime objects
            df_predictions.set_index('Tanggal', inplace=True)  # Set date as index
            
            # Merge with historical data
            df_merged = df_historical.merge(df_predictions, how='outer', left_index=True, right_index=True, suffixes=('_historical', '_predicted'))
            
            # Display results as a table
            st.write('Hasil prediksi:')
            st.dataframe(df_merged)  # Menampilkan DataFrame sebagai tabel
            
            # Display results as a line chart
            st.line_chart(df_merged[['Terakhir', 'predictions']])
    else:
        st.write('Silakan masukkan tanggal mulai dan tanggal akhir.')
