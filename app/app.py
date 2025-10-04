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

# Load final model artifacts
models = {}
try:
    # A3 Final Model
    with open(os.path.join(parent_dir, 'model_artifacts.pkl'), 'rb') as f:
        a3_data = pickle.load(f)
        models['A3'] = a3_data
        print(f"✅ A3 Final Model loaded: Classification model")
        
except Exception as e:
    print(f"Error loading model: {e}")

# Load dataset
try:
    data = pd.read_csv(os.path.join(parent_dir, 'Cars.csv'))
    print(f"✅ Dataset loaded: {len(data)} records")
except:
    data = pd.DataFrame()

# Model comparison data (updated with final results)
model_comparison = pd.DataFrame({
    'Assignment': ['A1', 'A2', 'A3'],
    'Model Type': ['Linear Regression', 'Enhanced Linear Regression', 'Logistic Classification'],
    'Problem Type': ['Regression', 'Regression', 'Classification'],
    'Best Score': ['R² = 0.6865', 'R² = 0.9091', 'Accuracy = 79.27%'],
    'Key Features': ['Basic implementation', 'Polynomial features + Lasso', 'Custom metrics + MLflow + CI/CD + Staging']
})

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Car Price Analysis - st126010"

# Styles
card_style = {
    'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '20px'
}

input_style = {
    'width': '100%', 'padding': '8px', 'borderRadius': '5px',
    'border': '1px solid #ddd', 'marginBottom': '10px'
}

button_style = {
    'backgroundColor': '#3498db', 'color': 'white', 'padding': '12px 24px',
    'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'width': '100%'
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🚗 Car Price Classification - Assignment 3", style={'textAlign': 'center', 'color': 'white'}),
        html.P("Student ID: st126010 - Htut Ko Ko", style={'textAlign': 'center', 'color': 'white'})
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'padding': '20px', 'marginBottom': '20px'
    }),
    
    # Tabs
    dcc.Tabs(id="tabs", value='comparison', children=[
        dcc.Tab(label='📊 Model Comparison', value='comparison'),
        dcc.Tab(label='🔮 Price Prediction', value='prediction'),
        dcc.Tab(label='📈 Data Analytics', value='analytics')
    ]),
    
    html.Div(id='tab-content', style={'padding': '20px'})
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f5f5'})

@callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_content(active_tab):
    if active_tab == 'comparison':
        return html.Div([
            html.Div([
                html.H2("Assignment Evolution & Results"),
                dash_table.DataTable(
                    data=model_comparison.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in model_comparison.columns],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white'},
                    style_data_conditional=[{
                        'if': {'row_index': 2}, 'backgroundColor': '#e8f5e8'
                    }]
                ),
                html.Div([dcc.Graph(figure=create_comparison_chart())], style={'marginTop': '20px'})
            ], style=card_style)
        ])
    
    elif active_tab == 'prediction':
        return html.Div([
            html.Div([
                html.H2("Car Price Prediction"),
                
                html.Div([
                    # Input form
                    html.Div([
                        html.H4("Select Model & Enter Details"),
                        
                        html.Label("Model:"),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[
                                {'label': 'A1 - Linear Regression', 'value': 'A1'},
                                {'label': 'A2 - Enhanced Linear Regression', 'value': 'A2'},
                                {'label': 'A3 - Logistic Classification', 'value': 'A3'}
                            ],
                            value='A1'
                        ),
                        
                        html.Label("Year:"),
                        dcc.Input(id='year-input', type='number', value=2015, style=input_style),
                        
                        html.Label("KM Driven:"),
                        dcc.Input(id='km-input', type='number', value=50000, style=input_style),
                        
                        html.Label("Fuel:"),
                        dcc.Dropdown(id='fuel-dropdown',
                                   options=[{'label': 'Petrol', 'value': 'Petrol'},
                                          {'label': 'Diesel', 'value': 'Diesel'},
                                          {'label': 'CNG', 'value': 'CNG'}],
                                   value='Petrol'),
                        
                        html.Label("Seller Type:"),
                        dcc.Dropdown(id='seller-dropdown',
                                   options=[{'label': 'Individual', 'value': 'Individual'},
                                          {'label': 'Dealer', 'value': 'Dealer'}],
                                   value='Individual'),
                        
                        html.Label("Transmission:"),
                        dcc.Dropdown(id='transmission-dropdown',
                                   options=[{'label': 'Manual', 'value': 'Manual'},
                                          {'label': 'Automatic', 'value': 'Automatic'}],
                                   value='Manual'),
                        
                        html.Label("Owner:"),
                        dcc.Dropdown(id='owner-dropdown',
                                   options=[{'label': 'First Owner', 'value': 'First Owner'},
                                          {'label': 'Second Owner', 'value': 'Second Owner'}],
                                   value='First Owner'),
                        
                        html.Label("Mileage:"),
                        dcc.Input(id='mileage-input', type='number', value=20, style=input_style),
                        
                        html.Label("Engine:"),
                        dcc.Input(id='engine-input', type='number', value=1200, style=input_style),
                        
                        html.Label("Max Power:"),
                        dcc.Input(id='power-input', type='number', value=80, style=input_style),
                        
                        html.Label("Seats:"),
                        dcc.Input(id='seats-input', type='number', value=5, style=input_style),
                        
                        html.Button('Predict Price', id='predict-button', n_clicks=0, style=button_style)
                        
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    # Results
                    html.Div([
                        html.H4("Prediction Result"),
                        html.Div(id='prediction-output', style={
                            'padding': '20px', 'backgroundColor': '#f8f9fa',
                            'borderRadius': '5px', 'minHeight': '200px'
                        })
                    ], style={'width': '48%', 'float': 'right'})
                    
                ], style={'overflow': 'hidden'})
            ], style=card_style)
        ])
    
    elif active_tab == 'analytics':
        return html.Div([
            html.Div([
                html.H2("Data Analytics"),
                html.Div([
                    html.Div([dcc.Graph(id='price-dist')], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(id='year-trend')], style={'width': '50%', 'display': 'inline-block'})
                ])
            ], style=card_style)
        ])

def create_comparison_chart():
    models_list = ['A1 Linear', 'A2 Enhanced', 'A3 Classification']
    scores = [0.9425, 0.8336, 0.7128]  # Updated scores
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models_list, y=scores,
        marker_color=['#e74c3c', '#f39c12', '#27ae60'],
        text=[f'{score:.4f}' for score in scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model', yaxis_title='Score',
        template='plotly_white'
    )
    return fig

@callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('model-dropdown', 'value'), Input('year-input', 'value'),
     Input('km-input', 'value'), Input('fuel-dropdown', 'value'),
     Input('seller-dropdown', 'value'), Input('transmission-dropdown', 'value'),
     Input('owner-dropdown', 'value'), Input('mileage-input', 'value'),
     Input('engine-input', 'value'), Input('power-input', 'value'),
     Input('seats-input', 'value')]
)
def predict_price(n_clicks, model_choice, year, km, fuel, seller, transmission, 
                 owner, mileage, engine, power, seats):
    if n_clicks == 0:
        return html.Div([
            html.P("Select model and enter car details, then click 'Predict Price'"),
            html.P("Available models:"),
            html.Ul([
                html.Li("A1: Basic Linear Regression"),
                html.Li("A2: Enhanced with Polynomial Features"),
                html.Li("A3: 4-Class Price Classification")
            ])
        ])
    
    try:
        if model_choice == 'A1':
            model_data = models['A1']
            input_data = pd.DataFrame({
                'name': ['Test Car'], 'year': [year], 'km_driven': [km], 'fuel': [fuel],
                'seller_type': [seller], 'transmission': [transmission], 'owner': [owner],
                'mileage': [f"{mileage} kmpl"], 'engine': [f"{engine} CC"], 
                'max_power': [f"{power} bhp"], 'torque': ['200Nm@ 2000rpm'], 'seats': [seats]
            })
            prediction = model_data['model'].predict(input_data)[0]
            
            return html.Div([
                html.H4("A1 Linear Regression Result", style={'color': '#e74c3c'}),
                html.H3(f"Predicted Price: {prediction:,.0f}", style={'color': '#27ae60'}),
                html.P(f"Model R² Score: {model_data['metrics']['r2']:.4f}"),
                html.P("✅ Prediction from A1 basic linear regression model")
            ])
            
        elif model_choice == 'A2':
            model_data = models['A2']
            
            # Manual encoding for A2 (match training order)
            fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
            seller_map = {'Individual': 0, 'Dealer': 1}
            transmission_map = {'Manual': 0, 'Automatic': 1}
            owner_map = {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3}
            
            features = np.array([[
                year, km, mileage, engine, power, seats,
                fuel_map.get(fuel, 0), seller_map.get(seller, 0),
                transmission_map.get(transmission, 0), owner_map.get(owner, 0)
            ]])
            
            features_scaled = model_data['scaler'].transform(features)
            features_poly = model_data['poly'].transform(features_scaled)
            raw_prediction = model_data['model'].predict(features_poly)[0]
            
            # Clip unrealistic predictions to reasonable range
            prediction = max(30000, min(10000000, abs(raw_prediction)))
            
            return html.Div([
                html.H4("A2 Enhanced Linear Regression Result", style={'color': '#f39c12'}),
                html.H3(f"Predicted Price: {prediction:,.0f}", style={'color': '#27ae60'}),
                html.P(f"Model R² Score: {model_data['test_r2']:.4f}"),
                html.P("✅ Enhanced with polynomial features and Lasso regularization"),
                html.P(f"Note: Raw prediction clipped to realistic range", style={'fontSize': '12px', 'color': '#666'})
            ])
            
        elif model_choice == 'A3':
            model_data = models['A3']
            
            # Manual encoding for A3
            fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
            seller_map = {'Individual': 0, 'Dealer': 1}
            transmission_map = {'Manual': 0, 'Automatic': 1}
            owner_map = {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3}
            
            features = np.array([[
                year, km, mileage, engine, power, seats,
                fuel_map.get(fuel, 0), seller_map.get(seller, 0),
                transmission_map.get(transmission, 0), owner_map.get(owner, 0)
            ]])
            
            features_scaled = model_data['scaler'].transform(features)
            prediction = model_data['model'].predict(features_scaled)[0]
            
            price_classes = {
                0: "Low (0 - 254999)",
                1: "Medium (254999 - 450000)", 
                2: "High (450000 - 675000)",
                3: "Premium (Above 675000)"
            }
            
            class_name = price_classes.get(int(prediction), "Unknown")
            
            return html.Div([
                html.H4("A3 Logistic Classification Result", style={'color': '#27ae60'}),
                html.H3(f"Price Class {int(prediction)}: {class_name}", style={'color': '#3498db'}),
                html.P("Model Accuracy: 71.28%"),
                html.P("✅ 4-class price classification with custom metrics")
            ])
            
    except Exception as e:
        return html.Div([
            html.H4("Prediction Error", style={'color': '#e74c3c'}),
            html.P(f"Error: {str(e)}"),
            html.P("Please check your inputs and try again.")
        ])

@callback(Output('price-dist', 'figure'), Input('tabs', 'value'))
def update_price_dist(tab):
    if data.empty:
        return go.Figure()
    
    fig = px.histogram(data, x='selling_price', nbins=30, 
                      title='Price Distribution')
    fig.update_layout(template='plotly_white')
    return fig

@callback(Output('year-trend', 'figure'), Input('tabs', 'value'))
def update_year_trend(tab):
    if data.empty:
        return go.Figure()
    
    yearly_avg = data.groupby('year')['selling_price'].mean().reset_index()
    fig = px.line(yearly_avg, x='year', y='selling_price', 
                 title='Average Price by Year', markers=True)
    fig.update_layout(template='plotly_white')
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
