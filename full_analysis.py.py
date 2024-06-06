import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Fungsi untuk membaca dan menampilkan data
def read_and_display_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan!")
    df = pd.read_csv(file_path)
    print(f"Data dari {file_path}:\n", df.head(), "\n")
    return df

# Membaca data dari file CSV
payment_df = read_and_display_csv('payment_data.csv')
production_df = read_and_display_csv('production_data.csv')
customer_df = read_and_display_csv('customer_data.csv')
product_df = read_and_display_csv('product_data.csv')
sales_df = read_and_display_csv('sales_data.csv')

# Memeriksa apakah kolom 'ProductID' ada di DataFrame sales_df
if 'ProductID' not in sales_df.columns:
    raise KeyError("Kolom 'ProductID' tidak ditemukan dalam data penjualan!")

# Transformasi data penjualan
sales_df['Date'] = pd.to_datetime(sales_df['Date'])
print("\nData penjualan setelah konversi tipe data:\n", sales_df.dtypes)

# EDA - Analisis Deskriptif
print("\nStatistik deskriptif data penjualan:\n", sales_df.describe())

# Visualisasi - Diagram Lingkaran (Pie Chart)
sales_per_product = sales_df.groupby('ProductID')['SaleAmount'].sum()
plt.figure(figsize=(8, 8))
plt.pie(sales_per_product, labels=sales_per_product.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribusi Penjualan per Produk')
plt.show()

# Visualisasi - Diagram Garis (Line Chart)
sales_over_time = sales_df.groupby('Date')['SaleAmount'].sum()
plt.figure(figsize=(12, 6))
plt.plot(sales_over_time.index, sales_over_time.values, marker='o')
plt.title('Tren Penjualan dari Waktu ke Waktu')
plt.xlabel('Tanggal')
plt.ylabel('Penjualan')
plt.grid(True)
plt.show()

# Visualisasi - Diagram Titik (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='SaleAmount', data=sales_df)
plt.title('Hubungan antara Kuantitas dan Penjualan')
plt.xlabel('Kuantitas')
plt.ylabel('Penjualan')
plt.grid(True)
plt.show()

# Modeling Data - Regresi Linear
X = sales_df[['Quantity']]
y = sales_df['SaleAmount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Validasi dan Tuning Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# Interpretasi dan Penyajian Hasil
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Quantity')
plt.ylabel('Sales')
plt.show()

# Deploy dan Monitoring - Menyimpan Model
joblib.dump(model, 'sales_model.pkl')

# Maintenance dan Iterasi - Memuat Model dan Membuat Prediksi Baru
loaded_model = joblib.load('sales_model.pkl')
new_prediction = loaded_model.predict([[20]])
print("\nPrediksi penjualan untuk Quantity 20:", new_prediction[0])

# Visualisasi tambahan - Total pembayaran per metode pembayaran
payment_method_totals = payment_df.groupby('PaymentMethod')['Amount'].sum()
plt.figure(figsize=(10, 6))
payment_method_totals.plot(kind='bar')
plt.title('Total Pembayaran per Metode Pembayaran')
plt.xlabel('Metode Pembayaran')
plt.ylabel('Total Pembayaran')
plt.show()

# Visualisasi tambahan - Produksi per produk
production_totals = production_df.groupby('ProductID')['QuantityProduced'].sum()
plt.figure(figsize=(10, 6))
production_totals.plot(kind='bar', color='green')
plt.title('Total Produksi per Produk')
plt.xlabel('Produk')
plt.ylabel('Total Produksi')
plt.show()

# Visualisasi tambahan - Penjualan per produk
sales_totals = sales_df.groupby('ProductID')['SaleAmount'].sum()
plt.figure(figsize=(10, 6))
sales_totals.plot(kind='bar', color='orange')
plt.title('Total Penjualan per Produk')
plt.xlabel('Produk')
plt.ylabel('Total Penjualan')
plt.show()

# Gabungkan DataFrame customer_df dan payment_df berdasarkan 'CustomerID'
merged_df = pd.merge(payment_df, customer_df, on='CustomerID')

# Tampilkan DataFrame yang sudah digabungkan
print("\nData gabungan customer dan payment:\n", merged_df.head())

# Visualisasi - Scatter Plot (Hubungan antara CustomerID dan Jumlah Pembayaran)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CustomerID', y='Amount', hue='Status', data=merged_df, palette='viridis')
plt.title('Hubungan antara CustomerID dan Jumlah Pembayaran')
plt.xlabel('CustomerID')
plt.ylabel('Jumlah Pembayaran')
plt.legend(title='Status Pembayaran')
plt.grid(True)
plt.show()
