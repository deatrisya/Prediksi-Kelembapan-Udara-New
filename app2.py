from flask import Flask, render_template, request, redirect, send_file, url_for
import pickle
# untuk memastikan file yang diunggah aman
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# library pdf generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import ParagraphStyle

import logging
logging.basicConfig(level=logging.DEBUG)

# import xlsxwriter
from io import BytesIO

import sys
sys.path.append('model')
from model_prediksi import NeuralNetwork, Sigmoid, Backpropagation

app = Flask(__name__)

def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def load_scalers():
    with open('scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    with open('preprocessing.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    return target_scaler, feature_scaler

@app.route('/main')
def main():
    return render_template('layouts/main.html')

@app.route('/')
def index():
    return render_template('home/index2.html',
                           
    type='tutorial')


def convert_to_label(input_value):
    if input_value == 0:
        return 8  # 'no clouds'
    elif input_value <= 10:
        return 0  # '10% or less, but not 0'
    elif input_value <= 30:
        return 2  # '20-30%'
    elif input_value <= 40:
        return 3  # '40%'
    elif input_value <= 50:
        return 4  # '50%'
    elif input_value <= 60:
        return 5  # '60%'
    elif input_value <= 80:
        return 6  # '70 - 80%'
    elif input_value < 100:
        return 7  # '90 or more, but not 100%'
    elif input_value == 100:
        return 1  # '100%'
    else:
        return None  # Menangani input yang tidak valid

def convert_to_label_excel(input_value):
    lower_input_value = input_value.lower()  # Ubah string menjadi huruf kecil
    
    if 'no clouds' in lower_input_value:
        return 8  # 'no clouds'
    elif '10%  or less, but not 0' in lower_input_value:
        return 0  # '10% or less, but not 0'
    elif '20-30%' in lower_input_value:
        return 2  # '20-30%'
    elif '40%.' in lower_input_value:
        return 3  # '40%'
    elif '50%.' in lower_input_value:
        return 4  # '50%'
    elif '60%.' in lower_input_value:
        return 5  # '60%'
    elif '70 - 80%.' in lower_input_value:
        return 6  # '70 - 80%'
    elif '90  or more, but not 100%' in lower_input_value:
        return 7  # '90 or more, but not 100%'
    elif '100%.' in lower_input_value:
        return 1  # '100%'
    else:
        return None 

def calculate_mse(y_test,y_pred):
    # Convert y_true and y_pred to array-like if they are not already
    y_test= np.array(y_test)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_test,y_pred)
    return mse

def calculate_mape(y_test,y_pred):
     # Convert y_true and y_pred to array-like if they are not already
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    mape = mean_absolute_percentage_error(y_test,y_pred)
    return mape

def generate_pdf_with_table(file_path, data):
    # Create a PDF document
    pdf = SimpleDocTemplate(file_path, pagesize=letter)
    
    # Table data - Replace this with your actual data
    table_data = data
    
    # Create the table
    num_cols = len(table_data[0])
    col_widths = [(letter[0] / num_cols) - 8] * num_cols
    
    # Create the table with full width
    table = Table(table_data, colWidths=col_widths)
    
    # Add style to the table
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),  # Header background color
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),       # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),              # Center alignment for all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),    # Header font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),             # Bottom padding for header row
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),     # Body background color
        ('GRID', (0, 0), (-1, -1), 1, colors.black)         # Gridlines
    ])
    table.setStyle(style)

    title_style = ParagraphStyle(name='TitleStyle', fontSize=18, textColor='navy', spaceAfter=20)
    
    # Create title paragraph
    title_paragraph = Paragraph("Prediksi Kelembapan Udara", style=title_style)
    
    
    # Add table to the PDF document
    pdf.build([title_paragraph, table])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model()
        target_scaler, feature_scaler = load_scalers()

        var_po = float(request.form['var_po'])
        var_n = int(request.form['var_n']) # tambahkan rentang
        var_vv = float(request.form['var_vv'])

        var_n_converted = convert_to_label(var_n)
        input_data = np.array([[var_po, var_n_converted, var_vv]])  # Menggabungkan nilai-nilai ke dalam array
        # Print statements for debugging
        print(f"Input Data (Original): {input_data}")
    
        normalized_input_data = feature_scaler.fit_transform(input_data)
        # Normalisasi data target
        # normalized_target_data = target_scaler.transform(np.array([[y_test]]).reshape(-1, 1))

        # Print statements for debugging
        print(f"Normalized Input Data: {normalized_input_data}")
        # print(f"Normalized Target Data: {normalized_target_data}")

        y_pred = model.predict_new_value(normalized_input_data)

        y_pred_denorm = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))[:, 0]  # Denormalisasi hasil prediksi

        # Print statements for debugging
        print(f"Predicted Normalized Output: {y_pred}")
        print(f"Denormalized Prediction: {y_pred_denorm}")

        return render_template('home/index2.html',
                            var_po = var_po,
                            var_n = var_n,
                            var_vv = var_vv,
                            prediction_text = y_pred_denorm[0],
                            type='manual',
                            success_message='berhasil melakukan prediksi')
    except Exception as e:
        return render_template('home/index2.html', errors=[str(e)], type='manual')

@app.route('/predict_from_excel', methods=['POST'])
def predict_from_excel():
    model = load_model()
    target_scaler, feature_scaler = load_scalers()

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    # Cek apakah file ada dan memiliki ekstensi yang diizinkan (xlsx atau csv)
    if file.filename == '' or not (file.filename.endswith('.xlsx') or file.filename.endswith('.csv')):
        return render_template('home/index2.html', 
            errors=['Format file harus .xlsx atau .csv'], 
            type='upload')

    if file.filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    elif file.filename.endswith('.csv'):
        data = pd.read_csv(file, encoding='latin1')
    
    original_values = []
    input_data = []
    # target_data = []
    predictions = []

    # Konversi nilai kolom VV ke numerik dan tangani nilai non-numerik
    data['VV'] = pd.to_numeric(data['VV'], errors='coerce')
    data['VV'] = data['VV'].replace({np.nan: 0.1})
    
    for index, row in data.iterrows():
        var_po = row['Po']
        var_n = row['N']
        var_vv = row['VV']
        # y_test = row['U']

        # Mengganti karakter yang tidak terbaca pada kolom 'N'
        if isinstance(var_n, str):
            var_n_cleaned = var_n.replace('', '-').replace('–', '-').replace('â', '-')
        else:
            var_n_cleaned = var_n
        
        original_values.append({'Po': var_po, 'N': var_n_cleaned, 'VV': var_vv})
        
        # Periksa tipe data kolom N
        if isinstance(var_n, str):
            if pd.isna(var_n):
                var_n_converted = np.nan
            var_n_converted = convert_to_label_excel(var_n)
        else:
            var_n_converted = convert_to_label(var_n)

        input_data.append([var_po, var_n_converted, var_vv]) 
        # target_data.append([y_test])
    

    # Imputasi data input
    input_imputer = KNNImputer(n_neighbors=5)
    input_data_imputed = input_imputer.fit_transform(input_data)
    # input_data[input_columns_numeric] = input_data_imputed

    # Imputasi data target
    # target_imputer = KNNImputer(n_neighbors=5)
    # target_data_imputed = target_imputer.fit_transform(target_data)
    # target_data[target_columns_numeric] = target_data_imputed

    normalized_input_data = feature_scaler.fit_transform(input_data_imputed)
    # Normalisasi data target
    # normalized_target_data = feature_scaler.fit_transform(target_data)


    for norm_input in zip(normalized_input_data):

        y_pred = model.predict_new_value(norm_input)
        
        y_pred_denorm = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))[:, 0]
        predictions.append(y_pred_denorm[0])

    # Hitung MSE menggunakan denormalisasi data
    # mse_test = calculate_mse(target_data_imputed,predictions)
    # Hitung MAPE menggunakan denormalisasi data
    # mape_test = calculate_mape(target_data_imputed,predictions)  

    # Generate Excel File
    excel_data = {
        key: [entry[key] for entry in original_values] for key in original_values[0]
    }
    excel_data['Prediksi'] = predictions

    df = pd.DataFrame(excel_data)
    df.to_excel('results/output.xlsx', index=False)

    # Generate PDF File
    pdf_data = [list(excel_data.keys())]  # Header row with dynamic keys
    num_rows = len(next(iter(excel_data.values())))

    for i in range(num_rows):
        row = []
        for key in excel_data.keys():
            row.append(excel_data[key][i])
        pdf_data.append(row)

    generate_pdf_with_table('results/output.pdf', pdf_data)

    return render_template('home/index2.html', predictions=predictions,
                            original_values=original_values,
                            # mse_excel = mse_test,
                            # mape_excel=f' { mape_test * 100:.2f} %',
                            type='upload',
                            success_message='berhasil melakukan prediksi')

@app.route('/download', methods=['GET'])
def download():
    # download by type
    if request.args.get('type') == 'pdf':
        return send_file('results/output.pdf', as_attachment=True, download_name='Hasil Prediksi Kelembapan Udara.pdf')
    elif request.args.get('type') == 'excel':
        return send_file('results/output.xlsx', as_attachment=True, download_name='Hasil Prediksi Kelembapan Udara.xlsx')

@app.route('/template', methods=['GET'])
def template():
    return send_file('static/template_nontarget.xlsx', as_attachment=True, download_name='Template Prediksi.xlsx')


if __name__ == '__main__':
    app.run(debug=True)