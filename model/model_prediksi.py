
#importing pandas library
import pandas as pd
import os

# Get the current working directory
current_dir = os.getcwd()

# Define the file path based on the operating system
if os.name == 'posix':  # Unix-based systems (like macOS and Linux)
    file_path = os.path.join(current_dir, 'model', 'weather.csv')
elif os.name == 'nt':   # Windows
    file_path = os.path.join(current_dir, 'model', 'weather.csv')
else:
    # Handle other operating systems if needed
    raise OSError("Unsupported operating system")
    
#read data from a file with path ('filename.csv')
df_latih = pd.read_csv(file_path, encoding='latin1')
#to view of the dataset
df_latih.head()

"""# PREPROCESSING DATA

## Menghapus Kolom yang Tidak Digunakan
"""

# Menghapus kolom yang tidak digunakan
kolom_tidak_digunakan = ["Local time in Surabaya", "ff10", "ff3", "WW", "W1", "W2", "Tn", "Tx", "RRR", "tR", "E", "Tg", "E'", "sss"]

# axis=1 digunakan untuk menunjukkan bahwa operasi penghapusan dilakukan pada kolom, bukan pada baris
df_latih = df_latih.drop(kolom_tidak_digunakan, axis=1)

# Menampilkan data berdasarkan kolom yang sudah di filter
df_latih.head(5)

"""## Memperbaiki Karakter yang Tidak Terbaca"""

# Kolom-kolom yang akan diubah
kolom_yang_akan_diubah = ['T', 'Po', 'P', 'Pa', 'U', 'DD',	'Ff',	'N',	'Cl',	'Nh',	'H',	'Cm',	'Ch',	'VV',	'Td']

# Iterasi melalui kolom-kolom yang akan diubah dan mengganti tanda  dengan tanda -
for kolom in kolom_yang_akan_diubah:
    df_latih[kolom] = df_latih[kolom].replace('', '-', regex=True)

# Menampilkan DataFrame setelah penggantian
df_latih.head()

"""## Perbaiki Data VV"""

#PENGECEKAN NILAI NON-NUMERIK

# Pilih kolom yang ingin diperiksa (misalnya, kolom "VV")
column_to_check = "VV"

# Konversi nilai kolom ke numerik
numeric_values = pd.to_numeric(df_latih[column_to_check], errors='coerce')

# Temukan baris yang memiliki nilai non-numerik
non_numeric_rows = df_latih[numeric_values.isna()]

# Tampilkan baris yang memiliki nilai non-numerik
print("\nBaris dengan nilai non-numerik:")
print(non_numeric_rows)

# Mengganti data 'less than 0.1'
df_latih[column_to_check] = df_latih[column_to_check].replace('less than 0.1', 1.0)

# Menampilkan Baris data setelah perbaikan
print("\nBaris VV Setelah Penggantian:")
# print(df_latih.loc[[5651, 6663]])

"""## Label Encoder



"""

from sklearn.preprocessing import LabelEncoder
import numpy as np

# Kolom kategorikal yang akan di encoded
kolom_kategorikal = ['DD', 'Ff', 'N', 'Cl', 'Nh', 'H', 'Cm', 'Ch']

# Membuat objek LabelEncoder yang mempertahankan nilai NaN
encoders = dict()  # Membuat dictionary kosong yang akan digunakan untuk menyimpan objek LabelEncoder untuk setiap kolom

for col_name in kolom_kategorikal:  # Loop hanya pada kolom kategorikal
    series = df_latih[col_name]  # Mengambil kolom yang sedang diproses dan menyimpannya dalam variabel series.
    label_encoder = LabelEncoder()
    df_latih[col_name] = pd.Series(
        label_encoder.fit_transform(series[series.notnull()]),  # Memilih hanya nilai yang tidak null dari kolom tersebut
        index=series[series.notnull()].index
    )
    encoders[col_name] = label_encoder

df_latih.head()

# Mengidentifikasi nilai kosong dalam DataFrame
nilai_kosong = df_latih.isnull().sum()
print(nilai_kosong)

"""## Memperbaiki missing value"""

#MEMPERBAIKI MISSING VALUE
from sklearn.impute import KNNImputer

columns_numeric = df_latih.columns

# Melakukan imputasi dengan KNNImputer pada kolom numerik
imputer_numeric = KNNImputer(n_neighbors=5)
df_latih[columns_numeric] = imputer_numeric.fit_transform(df_latih[columns_numeric])

# Menampilkan dataset setelah imputasi
print("\nDataFrame Setelah Imputasi:")
print(df_latih)
# Mengidentifikasi nilai kosong dalam DataFrame
nilai_kosong = df_latih.isnull().sum()
print(nilai_kosong)

"""## Normalisasi Min-Max Scaling

> Normalisasi Min-Max Scaling dilakukan untuk menyesuaikan skala semua fitur dalam dataset ke dalam rentang [0, 1].



`Rumus Min-Max Scaling:`

$$[ X_{\text{new}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}} ]$$

Di mana:
- $(X_{\text{scaled}})$ adalah nilai yang sudah dinormalisasi (skalanya berada dalam rentang [0, 1]).
- $(X)$ adalah nilai asli dari dataset.
- $(X_{\text{min}})$ adalah nilai minimum dari dataset.
- $(X_{\text{max}}$) adalah nilai maksimum dari dataset.

"""

from sklearn.preprocessing import MinMaxScaler

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(minmax_scaler.fit_transform(df_latih), columns=df_latih.columns)

# Menampilkan hasil normalisasi
print("Min-Max Scaled Data Latih:")
print(df_normalized.head())

"""## Train Test Split"""

from sklearn.model_selection import train_test_split

# Definisi fitur (X) dan target (y)
X = df_normalized.drop('U', axis=1)  # Menghapus kolom target 'U'yaitu Kelembapan Udara
y = df_normalized['U']  # Memilih kolom target 'U'


# Train-test split dengan rasio 80% pelatihan dan 20% pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menampilkan ukuran setiap bagian
print("Jumlah data pelatihan:", len(X_train))
print("Jumlah data pengujian:", len(X_test))

print("Ukuran X_train:", X_train.shape)
print("Ukuran y_train:", y_train.shape)
print("Ukuran X_test:", X_test.shape)
print("Ukuran y_test:", y_test.shape)

# Menampilkan data latih
print("Data Latih:")
print(X_train)
print("\n",y_train)

# Menampilkan data uji
print("\nData Uji:")
print(X_test)
print("\n",y_test)

"""# Library Backpropagation

## Class Neural Network
"""

import numpy as np

class NeuralNetwork:

    def __init__(self,input,hidden,output):
        self.input = input
        self.hidden = hidden
        self.output = output

    def initialize_weights(self, bias=False):
        self.hidden_weights=np.random.uniform(size=(self.input,self.hidden))
        self.output_weights=np.random.uniform(size=(self.hidden,self.output))
        self.bias = False
        if bias:
            self.hidden_bias_weights=np.random.uniform(size=(1,self.hidden))
            self.output_bias_weights=np.random.uniform(size=(1,self.output))
            self.bias = True

    def print_weights(self):
        print("Hidden Weights:")
        print(self.hidden_weights)
        print("\nOutput Weights:")
        print(self.output_weights)

        if self.bias:
            print("\nHidden Bias Weights:")
            print(self.hidden_bias_weights)
            print("\nOutput Bias Weights:")
            print(self.output_bias_weights)

"""## Class Sigmoid"""

import numpy as np

class Sigmoid:
    def activate(self, x):
        return 1/(1 + np.exp(-x))
    def derivative(self, x):
        return x * (1 - x)

"""## Class Feedforward Backward"""

class Backpropagation:

    def __init__(self, neuralnet, epochs=2000, lr=0.1, activation_function=Sigmoid()):
        self.neuralnet = neuralnet
        self.epochs = epochs
        self.lr = lr
        self.activation_function = activation_function

    def feedForward(self, input):
        hidden_layer = np.dot(input, self.neuralnet.hidden_weights)
        if self.neuralnet.bias:
            hidden_layer += self.neuralnet.hidden_bias_weights
        hidden_layer = self.activation_function.activate(hidden_layer)

        output_layer = np.dot(hidden_layer, self.neuralnet.output_weights)
        if self.neuralnet.bias:
            output_layer += self.neuralnet.output_bias_weights
        output_layer = self.activation_function.activate(output_layer)

        # Print hidden_layer and output_layer
        # print("\n Input Layer:\n", input)
        # print("Hidden Layer (Before Activation):\n", np.dot(input, self.neuralnet.hidden_weights))
        # print("Hidden Layer (After Activation):\n", hidden_layer)
        # print("Output Layer (Before Activation):\n", np.dot(hidden_layer, self.neuralnet.output_weights))
        # print("Output Layer (After Activation):\n", output_layer)
        return hidden_layer, output_layer

    def train(self, input, target):
        for epoch in range(self.epochs):
            # print(f"\nEpoch {epoch + 1}/{self.epochs}:")

            # Feed Forward
            hidden_layer, output_layer = self.feedForward(input)

            # Error term for each output unit k
            derivative_output = self.activation_function.derivative(output_layer)
            del_k = output_layer * derivative_output * (target - output_layer)

            # Error term for each hidden unit h
            sum_del_h = del_k.dot(self.neuralnet.output_weights.T)
            derivative_hidden = self.activation_function.derivative(hidden_layer)
            del_h = hidden_layer * derivative_hidden * sum_del_h

            # Print values
            # print("Derivative output for each output unit k (del_k):\n", derivative_output)
            # print("\nError term for each output unit k (del_k):\n", del_k)

            # Print values
            # print("Jumlah delta for each hidden unit h (del_h):\n", sum_del_h)
            # print("Derivative hidden for each hidden unit h (del_h):\n", derivative_hidden)
            # print("Error term for each hidden unit h (del_h):\n", del_h)

            # Weight Update
            self.neuralnet.output_weights += hidden_layer.T.dot(del_k) * self.lr
            self.neuralnet.hidden_weights += input.T.dot(del_h) * self.lr

            # Print updated weights
            # print("\nUpdated Output Weights:\n", self.neuralnet.output_weights)
            # print("Updated Hidden Weights:\n", self.neuralnet.hidden_weights)

    def predict(self, input, actual_output):
        hidden_layer, output_layer = self.feedForward(input)
        predicted_values = []  # List untuk menyimpan nilai output_layer[i][0]

        for i in range(len(input)):
          for j in range(len(actual_output)):
            if i==j:
              predicted_value = output_layer[i][j]
              actual_value = actual_output[i][0]

              # print(f"For input {input[i]}, the predicted output is {predicted_value} and the actual output is {actual_value}")

              # Simpan nilai predicted_value ke dalam list
              predicted_values.append(predicted_value)

        return predicted_values
    
    def predict_new_value(self, input):
        hidden_layer, output_layer = self.feedForward(input)
        predicted_values = []  # List untuk menyimpan nilai output_layer[i][0]
        for i in range(len(input)):
            for j in range(len(input)):
                if i==j:
                    predicted_value = output_layer[i][j]
                # print(f"For input {input[i]}, the predicted output is {predicted_value} and the actual output is {actual_value}")
                # Simpan nilai predicted_value ke dalam list
            predicted_values.append(predicted_value)
        return predicted_values
"""# Backpropagation

## Seleksi Fitur

## Fitur Terpilih
"""

best_solution = [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]

# Fungsi seleksi fitur
def select_features(solution, df):
    selected_columns = [col for col, value in zip(df.columns, solution[0]) if value == 1]
    selected_df = df[selected_columns]
    return selected_df

# Pilih fitur-fitur yang sesuai dengan best solution
selected_features_test = select_features(best_solution, X_test)
print('Variabel Sub Fitur Terpilih')
print(selected_features_test.columns)

print('\nselected_features_test')
print(selected_features_test)

"""## Prediksi Data Uji"""

import time
import pickle
t1 = time.perf_counter()

# Inisialisasi objek NeuralNetwork
input_size = selected_features_test.shape[1]
hidden_size=2
output_size=len(y_test)
nn = NeuralNetwork(input_size, hidden_size, output_size)

#Print jumlah neuron tiap layer
# print("input size", input_size)
# print("hidden size", hidden_size)
# print("output size", output_size, "\n")

# Inisialisasi bobot dengan atau tanpa bias
nn.initialize_weights(bias=True)  # Sesuaikan dengan kebutuhan

# Cetak bobot
# nn.print_weights()

# Inisialisasi objek Backpropagation dengan objek NeuralNetwork yang telah dibuat
epochs=5

learning_rate=0.005
activation_function=Sigmoid()
bp = Backpropagation(nn, epochs, learning_rate, activation_function)

# Latih model dengan data uji
input_data = selected_features_test.values
target_data = np.array(y_test)

# Lakukan prediksi dengan data uji
y_pred = bp.predict(input_data,target_data.reshape(-1,1))
# y_pred   = bp.predict(input_data)

#Denormalisasi
scaler_U = MinMaxScaler() # Membuat skalar baru untuk kolom U
scaler_U.fit(df_latih['U'].values.reshape(-1, 1)) # Pasangkan scaler baru dengan nilai asli 'U'
y_pred_denorm = scaler_U.inverse_transform(np.array(y_pred).reshape(-1, 1))[:, 0] # Denormalisasi kolom 'U'
y_test_denorm = scaler_U.inverse_transform(np.array(y_test).reshape(-1, 1))[:, 0]

# # Membuat DataFrame untuk menampung hasil
df_results_test = pd.DataFrame({
    'Nilai Prediksi': y_pred,
    'Nilai Target': y_test,
    'Denormalisasi Nilai Prediksi': y_pred_denorm,
    'Denormalisasi Nilai Target': y_test_denorm
})
   # print('\nPerbandingan Hasil Normalisasi & Denormalisasi')
print(df_results_test)


t2 = time.perf_counter()
print('Waktu yang dibutuhkan untuk eksekusi', t2-t1, 'detik')

"""# Evaluasi MSE

### Data Uji
"""

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# # Menghitung MSE data uji denormalisasi
mse_test = mean_squared_error(y_test_denorm, y_pred_denorm)
print("Mean Squared Error:", mse_test)

# Menghitung MSE data uji normalisasi
# mse_test = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse_test)

# """# Evaluasi MAPE

# ### Data Uji
# """

# Menghitung MAPE data uji denormalisasi
mape_test = mean_absolute_percentage_error(y_test_denorm, y_pred_denorm)
mape_test_percent = round(mape_test,2)
print("Mean Absolute Percentage Error:", mape_test)
print("Mean Absolute Percentage Error (in percent):", mape_test_percent,"%")

# Menghitung MAPE data uji normalisasi
# mape_test = mean_absolute_percentage_error(y_test, y_pred)
# mape_test_percent = round(mape_test,2)
# print("Mean Absolute Percentage Error:", mape_test)
# print("Mean Absolute Percentage Error (in percent):", mape_test_percent,"%")

"""# MODEL"""

# Simpan model ke file menggunakan pickle
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(bp, f)

# # Simpan scaler ke file menggunakan pickle
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler_U, f)

# # Simpan scaler ke file menggunakan pickle
# with open('preprocessing.pkl', 'wb') as f:
#     pickle.dump(minmax_scaler, f)

# with open('neuralnetwork.pkl', 'wb') as f:
#     pickle.dump(nn, f)