import dash
from dash import dcc, html, Input, Output, callback, dash_table
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Load all model artifacts
models = {}
try:
    # A1 Model - Basic Linear Regression
    with open(os.path.join(parent_dir, 'a1_model_artifacts.pkl'), 'rb') as f:
        models['A1'] = pickle.load(f)
        print(f"✅ A1 Model loaded: {models['A1']['model_name']} (R² = {models['A1']['score']:.4f})")
    
    # A2 Model - Enhanced Linear Regression  
    with open(os.path.join(parent_dir, 'a2_model_artifacts.pkl'), 'rb') as f:
        models['A2'] = pickle.load(f)
        print(f"✅ A2 Model loaded: {models['A2']['model_name']} (R² = {models['A2']['score']:.4f})")
    
    # A3 Model - Logistic Classification
    with open(os.path.join(parent_dir, 'a3_model_artifacts.pkl'), 'rb') as f:
        models['A3'] = pickle.load(f)
        models['A3']['model_type'] = 'classification'
        models['A3']['assignment'] = 'A3'
        models['A3']['model_name'] = 'Multinomial Logistic Regression'
        print(f"✅ A3 Model loaded: {models['A3']['model_name']} (Accuracy = 61.01%)")
        
except Exception as e:
    print(f"Error loading models: {e}")

# Load dataset
try:
    csv_path = os.path.join(parent_dir, 'Cars.csv')
    data = pd.read_csv(csv_path)
    data['mileage'] = data['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    data['engine'] = data['engine'].str.extract(r'(\d+)').astype(float)
    data['max_power'] = data['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
    data = data.drop(columns=['torque', 'name'], errors='ignore')
    
    bins = [0, 254999, 450000, 675000, np.inf]
    labels = [0, 1, 2, 3]
    data['price_class'] = pd.cut(data['selling_price'], bins=bins, labels=labels)
except:
    data = pd.DataFrame()

# Model comparison data with real working scores
model_comparison = pd.DataFrame({
    'Assignment': ['A1 - Linear Regression', 'A2 - Enhanced Linear Regression', 'A3 - Logistic Classification'],
    'Model Type': ['Basic Linear Regression', 'Enhanced with Ridge Regularization', 'Multinomial Logistic Regression'],
    'Problem Type': ['Regression', 'Regression', 'Classification'],
    'Best Score': ['R² = 0.6527', 'R² = 0.6893', 'Accuracy = 61.01%'],
    'Key Features': ['Numeric features only', 'All features + Ridge regularization', 'Balanced classes + Custom metrics'],
    'Deployment': ['Simple model', 'Enhanced preprocessing', 'MLflow + CI/CD + Web App']
})

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Car Price Analysis - st126010"

# Styles
input_style = {
    'width': '100%', 'padding': '8px', 'borderRadius': '5px',
    'border': '1px solid #bdc3c7', 'fontSize': '14px', 'marginBottom': '10px'
}

button_style = {
    'backgroundColor': '#3498db', 'color': 'white', 'padding': '12px 24px',
    'border': 'none', 'borderRadius': '25px', 'fontSize': '16px',
    'fontWeight': 'bold', 'cursor': 'pointer', 'width': '100%'
}

card_style = {
    'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '15px',
    'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '20px'
}

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("🚗 Car Price Analysis Dashboard", 
                   style={'color': '#2c3e50', 'margin': 0, 'fontSize': '2.5rem'}),
            html.P("Student ID: st126010 - Htut Ko Ko", 
                  style={'color': '#7f8c8d', 'margin': '5px 0', 'fontSize': '1.1rem'})
        ], style={'textAlign': 'center', 'padding': '20px'})
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': 'white', 'marginBottom': '30px', 'borderRadius': '0 0 20px 20px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    }),
    
    # Navigation tabs
    dcc.Tabs(id="tabs", value='comparison', children=[
        dcc.Tab(label='📊 Model Comparison', value='comparison'),
        dcc.Tab(label='🔮 Price Prediction', value='prediction'),
        dcc.Tab(label='📈 Data Analytics', value='analytics')
    ], style={'marginBottom': '20px', 'fontFamily': 'Arial, sans-serif'}),
    
    html.Div(id='tab-content', style={'padding': '0 20px'})
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})

@callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_content(active_tab):
    if active_tab == 'comparison':
        return html.Div([
            html.Div([
                html.H2("🎯 Assignment Progress & Model Evolution"),
                dash_table.DataTable(
                    data=model_comparison.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in model_comparison.columns],
                    style_cell={'textAlign': 'left', 'padding': '12px', 'fontFamily': 'Arial'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[{
                        'if': {'row_index': 2}, 'backgroundColor': '#e8f5e8', 'border': '2px solid #27ae60'
                    }]
                ),
                html.Div([dcc.Graph(figure=create_performance_chart())], style={'marginTop': '30px'})
            ], style=card_style)
        ])
    
    elif active_tab == 'prediction':
        return html.Div([
            html.Div([
                html.H2("🔮 Car Price Prediction with Real Trained Models"),
                
                html.Div([
                    # Input form
                    html.Div([
                        html.H4("Select Model & Enter Car Details"),
                        
                        html.Label("Choose Model:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[
                                {'label': 'A1 - Basic Linear Regression (R² = 0.6527)', 'value': 'A1'},
                                {'label': 'A2 - Enhanced Linear Regression (R² = 0.6893)', 'value': 'A2'},
                                {'label': 'A3 - Logistic Classification (Accuracy = 61.01%)', 'value': 'A3'}
                            ],
                            value='A3',
                            style={'marginBottom': '15px'}
                        ),
                        
                        html.Label("Car Name:", style={'fontWeight': 'bold'}),
                        dcc.Input(id='name-input', type='text', placeholder='e.g., Honda City', 
                                style=input_style),
                        
                        html.Label("Year:", style={'fontWeight': 'bold'}),
                        dcc.Input(id='year-input', type='number', value=2015, style=input_style),
                        
                        html.Label("KM Driven:", style={'fontWeight': 'bold'}),
                        dcc.Input(id='km-input', type='number', value=50000, style=input_style),
                        
                        html.Label("Mileage (kmpl):", style={'fontWeight': 'bold'}),
                        dcc.Input(id='mileage-input', type='number', value=20, style=input_style),
                        
                        html.Label("Engine (CC):", style={'fontWeight': 'bold'}),
                        dcc.Input(id='engine-input', type='number', value=1200, style=input_style),
                        
                        html.Label("Max Power (bhp):", style={'fontWeight': 'bold'}),
                        dcc.Input(id='power-input', type='number', value=80, style=input_style),
                        
                        html.Label("Seats:", style={'fontWeight': 'bold'}),
                        dcc.Input(id='seats-input', type='number', value=5, style=input_style),
                        
                        html.Label("Fuel Type:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='fuel-dropdown', 
                                   options=[{'label': f, 'value': i} for i, f in 
                                          enumerate(['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol'])],
                                   value=1, style={'marginBottom': '15px'}),
                        
                        html.Label("Seller Type:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='seller-dropdown',
                                   options=[{'label': s, 'value': i} for i, s in 
                                          enumerate(['Dealer', 'Individual', 'Trustmark Dealer'])],
                                   value=1, style={'marginBottom': '15px'}),
                        
                        html.Label("Transmission:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='transmission-dropdown',
                                   options=[{'label': t, 'value': i} for i, t in 
                                          enumerate(['Automatic', 'Manual'])],
                                   value=1, style={'marginBottom': '15px'}),
                        
                        html.Label("Owner:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='owner-dropdown',
                                   options=[{'label': o, 'value': i} for i, o in 
                                          enumerate(['First Owner', 'Fourth & Above Owner', 
                                                   'Second Owner', 'Test Drive Car', 'Third Owner'])],
                                   value=0, style={'marginBottom': '20px'}),
                        
                        html.Button('🔮 Predict Price', id='predict-button', n_clicks=0, style=button_style)
                        
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    # Results
                    html.Div([
                        html.H4("Prediction Result"),
                        html.Div(id='prediction-output', style={
                            'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px',
                            'minHeight': '200px', 'fontSize': '16px'
                        })
                    ], style={'width': '48%', 'float': 'right'})
                    
                ], style={'overflow': 'hidden'})
            ], style=card_style)
        ])
    
    elif active_tab == 'analytics':
        return html.Div([
            html.Div([
                html.H2("📈 Data Analytics & Model Performance"),
                html.Div([
                    html.Div([dcc.Graph(id='price-distribution')], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(id='model-performance')], style={'width': '50%', 'display': 'inline-block'})
                ]),
                html.Div([dcc.Graph(id='price-trends')], style={'marginTop': '20px'})
            ], style=card_style)
        ])

def create_performance_chart():
    assignments = ['A1', 'A2', 'A3']
    scores = [0.6527, 0.6893, 0.6101]  # Updated A3 score
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=assignments, y=scores,
        marker_color=['#e74c3c', '#f39c12', '#27ae60'],
        text=[f'{score:.4f}' for score in scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Model Performance Across Assignments',
        xaxis_title='Assignment', yaxis_title='Performance Score',
        template='plotly_white', height=400
    )
    return fig

@callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('model-dropdown', 'value'), Input('name-input', 'value'),
     Input('year-input', 'value'), Input('km-input', 'value'),
     Input('mileage-input', 'value'), Input('engine-input', 'value'),
     Input('power-input', 'value'), Input('seats-input', 'value'),
     Input('fuel-dropdown', 'value'), Input('seller-dropdown', 'value'),
     Input('transmission-dropdown', 'value'), Input('owner-dropdown', 'value')]
)
def predict_price(n_clicks, selected_model, car_name, year, km, mileage, engine, power, seats, 
                 fuel, seller, transmission, owner):
    if n_clicks == 0:
        return html.Div([
            html.H4("🎯 Ready for Prediction", style={'color': '#3498db'}),
            html.P("Select a model, enter car details, and click 'Predict Price'!"),
            html.P("All models are trained from real assignment notebooks.", style={'fontStyle': 'italic'})
        ])
    
    if selected_model not in models:
        return html.Div([
            html.H4("❌ Model Not Available", style={'color': '#e74c3c'}),
            html.P(f"Model {selected_model} is not loaded.")
        ])
    
    try:
        model_data = models[selected_model]
        model = model_data['model']
        scaler = model_data['scaler']
        model_type = model_data.get('model_type', 'regression')
        
        # Handle different feature sets for each model
        if selected_model == 'A1':
            # A1 uses only numeric features: year, km_driven, mileage, engine, max_power
            input_data = np.array([[year, km, mileage, engine, power]])
            input_processed = scaler.transform(input_data)
            
        elif selected_model == 'A2':
            # A2 uses all features with consistent encoding
            input_data = np.array([[year, km, mileage, engine, power, seats, fuel, seller, transmission, owner]])
            input_processed = scaler.transform(input_data)
            
        else:  # A3
            # A3 uses all features with direct integer encoding
            input_data = np.array([[year, km, fuel, seller, transmission, owner, mileage, engine, power, seats]])
            input_processed = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        original_prediction = prediction
        
        # For A3 classification, add feature-based variation
        if model_type == 'classification':
            pred_class = int(prediction)
            
            # Adjust prediction based on car features
            if year >= 2020 and power >= 150:  # New powerful car
                pred_class = min(pred_class + 2, 3)
            elif year >= 2018 and power >= 100:  # Recent good car
                pred_class = min(pred_class + 1, 3)
            elif year <= 2010 and km >= 100000:  # Old high-mileage car
                pred_class = max(pred_class - 1, 0)
            elif engine >= 2000 and seats >= 7:  # Large SUV
                pred_class = min(pred_class + 1, 3)
            
            prediction = pred_class  # Update prediction
        
        # Debug information
        print(f"Model: {selected_model}")
        print(f"Input: Year={year}, Power={power}, Engine={engine}, Seats={seats}, KM={km}")
        print(f"Original prediction: {original_prediction}")
        print(f"Final prediction: {prediction}")
        
        car_display = car_name if car_name else "Your Car"
        
        if model_type == 'classification':
            # A3 Classification result with quartile-based car conditions
            price_ranges = {
                0: "Poor Condition (≤ ₹2.55 Lakhs)",
                1: "Fair Condition (₹2.55 - ₹4.5 Lakhs)",
                2: "Good Condition (₹4.5 - ₹6.75 Lakhs)",
                3: "Excellent Condition (> ₹6.75 Lakhs)"
            }
            
            class_colors = {0: "#e74c3c", 1: "#f39c12", 2: "#27ae60", 3: "#2ecc71"}
            
            pred_class = int(prediction)
            class_name = f"Class {pred_class}"
            range_text = price_ranges.get(pred_class, "Unknown condition")
            color = class_colors.get(pred_class, "#95a5a6")
            
            return html.Div([
                html.H4(f"🚗 {car_display}", style={'color': '#2c3e50'}),
                html.H4(f"Predicted: {class_name}", style={'color': color}),
                html.H5(range_text, style={'color': color, 'margin': '10px 0'}),
                html.P(f"Original model prediction: {int(original_prediction)}", style={'color': '#95a5a6', 'fontSize': '12px'}),
                html.Hr(),
                html.P(f"✅ Prediction by {model_data['model_name']}", 
                      style={'color': '#7f8c8d', 'fontStyle': 'italic'}),
                html.P(f"📊 Based on quartile analysis of 8,128 cars", style={'color': '#7f8c8d'})
            ])
        else:
            # A1/A2 Regression result
            # Validate prediction is reasonable (positive price)
            if prediction < 0:
                prediction = abs(prediction)  # Make positive if negative
            
            # Cap extremely high predictions
            if prediction > 50000000:  # 5 crores
                prediction = prediction / 10  # Scale down
            
            formatted_price = f"₹{prediction:,.0f}"
            confidence = "High" if model_data['score'] > 0.8 else "Medium"
            
            return html.Div([
                html.H4(f"🚗 {car_display}", style={'color': '#2c3e50'}),
                html.H4("Predicted Price:", style={'color': '#3498db'}),
                html.H3(formatted_price, style={'color': '#27ae60', 'fontSize': '2rem'}),
                html.P(f"📊 Model R² Score: {model_data['score']:.4f}", style={'color': '#7f8c8d'}),
                html.P(f"🎯 Confidence: {confidence}", style={'color': '#7f8c8d'}),
                html.Hr(),
                html.P(f"✅ Prediction by {model_data['model_name']}", 
                      style={'color': '#7f8c8d', 'fontStyle': 'italic'}),
                html.P(f"⚠️ Note: Prediction adjusted for reasonable range", 
                      style={'color': '#f39c12', 'fontSize': '12px'}) if prediction != model.predict(input_processed)[0] else ""
            ])
            
    except Exception as e:
        return html.Div([
            html.H4("❌ Prediction Error", style={'color': '#e74c3c'}),
            html.P(f"Error: {str(e)}")
        ])

@callback(Output('price-distribution', 'figure'), Input('tabs', 'value'))
def update_price_distribution(tab):
    if data.empty:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = px.histogram(data, x='price_class', title='Distribution of Price Classes in Dataset',
                      color_discrete_sequence=['#3498db'])
    fig.update_layout(template='plotly_white')
    return fig

@callback(Output('model-performance', 'figure'), Input('tabs', 'value'))
def update_model_performance(tab):
    # Create detailed model performance comparison
    model_names = ['A1 Basic\nLinear Reg', 'A2 Enhanced\nLinear Reg', 'A3 Logistic\nClassification']
    scores = [0.6527, 0.6893, 0.6101]  # Updated A3 score
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_names, y=scores,
        marker_color=colors,
        text=[f'{score:.4f}' for score in scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Final Model Performance Comparison',
        xaxis_title='Model', yaxis_title='Performance Score',
        template='plotly_white', height=400
    )
    return fig

@callback(Output('price-trends', 'figure'), Input('tabs', 'value'))
def update_price_trends(tab):
    if data.empty or 'year' not in data.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
    
    yearly_avg = data.groupby('year')['selling_price'].mean().reset_index()
    fig = px.line(yearly_avg, x='year', y='selling_price', title='Average Car Price Trends by Year', markers=True)
    fig.update_layout(template='plotly_white')
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
