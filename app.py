import pickle
import gradio as gr
import numpy as np
default_values_array = [0, 7.215307, 0.339666, 0.318633, 5.443235, 0.056034, 30.525319, 115.744574, 0.994697, 3.218501, 0.531268, 10.491801]
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

def predict_quality(type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                    pH, sulphates, alcohol):
    type_value = 0 if type == "Red" else 1
    input_data = np.array([type_value, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                           chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                           pH, sulphates, alcohol]).reshape(1, -1)
    prediction = loaded_model.predict(input_data)[0]
    quality = "Wine Quality is Good" if prediction >= 7 else "Wine Quality is Bad"
    return quality, prediction

def predict_quality_array(input_array):
    try:
        input_data = np.array([float(x.strip()) for x in input_array.split(',')]).reshape(1, -1)
        if input_data.shape[1] != 12:
            return "Error: Input should contain exactly 12 values", None
        prediction = loaded_model.predict(input_data)[0]
        quality = "Wine Quality is Good" if prediction >= 7 else "Wine Quality is Bad"
        return quality, prediction
    except ValueError:
        return "Error: Invalid input. Please enter numeric values separated by commas.", None

with gr.Blocks() as interface:
    gr.Markdown("# Wine Quality Prediction")
    
    with gr.Tab("Individual Inputs"):
        type = gr.Radio(["Red", "White"], label="Type", value="Red" if default_values_array[0] == 0 else "White")
        fixed_acidity = gr.Number(label="Fixed Acidity", value=default_values_array[1])
        volatile_acidity = gr.Number(label="Volatile Acidity", value=default_values_array[2])
        citric_acid = gr.Number(label="Citric Acid", value=default_values_array[3])
        residual_sugar = gr.Number(label="Residual Sugar", value=default_values_array[4])
        chlorides = gr.Number(label="Chlorides", value=default_values_array[5])
        free_sulfur_dioxide = gr.Number(label="Free Sulfur Dioxide", value=default_values_array[6])
        total_sulfur_dioxide = gr.Number(label="Total Sulfur Dioxide", value=default_values_array[7])
        density = gr.Number(label="Density", value=default_values_array[8])
        pH = gr.Number(label="pH", value=default_values_array[9])
        sulphates = gr.Number(label="Sulphates", value=default_values_array[10])
        alcohol = gr.Number(label="Alcohol", value=default_values_array[11])
        
        predict_button = gr.Button("Predict")
        
        quality_output = gr.Textbox(label="Wine Quality")
        prediction_output = gr.Number(label="Predicted Value")
        
        predict_button.click(
            predict_quality,
            inputs=[type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                    pH, sulphates, alcohol],
            outputs=[quality_output, prediction_output]
        )
    
    with gr.Tab("Array Input"):
        array_input = gr.Textbox(
            label="Input Array",
            placeholder="Enter 12 comma-separated values: type (0 for Red, 1 for White), fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol"
        )
        array_predict_button = gr.Button("Predict")
        
        array_quality_output = gr.Textbox(label="Wine Quality")
        array_prediction_output = gr.Number(label="Predicted Value")
        
        array_predict_button.click(
            predict_quality_array,
            inputs=array_input,
            outputs=[array_quality_output, array_prediction_output]
        )

interface.launch()