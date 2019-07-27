from fbprophet import Prophet
import pandas as pd

dfs = pd.read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20180101&end=20181231')
df = dfs[0]
#print(df.head())

df = df.iloc[::-1] #시간순으로 뒤집기
#print(df.head())


df['ds'] = df['Date']
df['y'] = df['Close**']
forecast_data = df[['ds','y']].copy()
print(forecast_data())

m = Prophet()
m.fit(forecast_data)

future = m.make_future_dataframe(periods=90)
print(future.tail())

forcast = m.predict(future)
print(forcast)

print('=2019-3-31 최저 최고값 예측값 ==')
print(forcast[['ds', 'yhat','yhat_lower','yhat_upper']].tail())


