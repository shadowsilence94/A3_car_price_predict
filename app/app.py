import dash
from dash import dcc, html, Input, Output, callback, dash_table
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    # A1 Model
    with open(os.path.join(parent_dir, 'a1_model_artifacts.pkl'), 'rb') as f:
        a1_data = pickle.load(f)
        models['A1'] = a1_data
        print(f"✅ A1 Model loaded: R² = {a1_data['metrics']['test_r2']:.4f}")
    
    # A2 Model
    with open(os.path.join(parent_dir, 'a2_model_artifacts.pkl'), 'rb') as f:
        a2_data = pickle.load(f)
        models['A2'] = a2_data
        print(f"✅ A2 Model loaded: R² = {a2_data['metrics']['test_r2']:.4f}")
    
    # A3 Model
    with open(os.path.join(parent_dir, 'a3_model_artifacts.pkl'), 'rb') as f:
        a3_data = pickle.load(f)
        models['A3'] = a3_data
        print(f"✅ A3 Model loaded: Classification model")
        
except Exception as e:
    print(f"Error loading models: {e}")

# Load dataset
try:
    data = pd.read_csv(os.path.join(parent_dir, 'Cars.csv'))
    print(f"✅ Dataset loaded: {len(data)} records")
except:
    data = pd.DataFrame()

# Model comparison data (aligned with current actual results)
model_comparison = pd.DataFrame({
    'Assignment': ['A1', 'A2', 'A3'],
    'Model Type': ['Linear Regression', 'Enhanced Linear Regression', 'Logistic Classification'],
    'Problem Type': ['Regression', 'Regression', 'Classification'],
    'Best Score': ['R² = 0.7657', 'R² = 0.9101', 'Accuracy = 74.05%'],
    'Key Features': ['Proper ML pipeline + log transform', 'Polynomial features + Lasso regularization', 'Custom logistic regression + Ridge penalty']
})

# Initialize app with external CSS
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
        'rel': 'stylesheet'
    }
]

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
app.title = "Car Price Analytics - st126010"

# Add responsive meta tag
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @media (max-width: 768px) {
                .mobile-full { width: 100% !important; display: block !important; margin-bottom: 20px !important; }
                .mobile-stack { flex-direction: column !important; }
                .mobile-text { font-size: 14px !important; }
                .mobile-padding { padding: 10px !important; }
            }
            .chart-container { height: 400px; }
            @media (max-width: 768px) {
                .chart-container { height: 300px; }
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                margin: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .tab-content { animation: fadeIn 0.5s; }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            body { font-family: 'Inter', sans-serif !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Responsive styles
mobile_style = {
    '@media (max-width: 768px)': {
        'width': '100% !important',
        'display': 'block !important',
        'marginBottom': '20px'
    }
}

card_style = {
    'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '12px',
    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'marginBottom': '20px',
    'border': '1px solid #e1e8ed'
}

input_style = {
    'width': '100%', 'padding': '10px', 'borderRadius': '8px',
    'border': '1px solid #ddd', 'marginBottom': '20px', 'fontSize': '14px'
}

button_style = {
    'backgroundColor': '#3498db', 'color': 'white', 'padding': '12px 24px',
    'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer', 'width': '100%',
    'fontSize': '16px', 'fontWeight': 'bold', 'transition': 'all 0.3s ease'
}

# Enhanced color palette
colors = {
    'primary': '#3498db',
    'secondary': '#2ecc71', 
    'accent': '#e74c3c',
    'warning': '#f39c12',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}

# Layout with responsive design
app.layout = html.Div([
    # Header with responsive design and better alignment
    html.Div([
        html.Div([
            html.H1("Car Price Analytics Dashboard", 
                   style={'margin': '0', 'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center'}),
            html.P("Advanced ML Models & Data Insights | st126010 - Htut Ko Ko", 
                  style={'margin': '5px 0 0 0', 'fontSize': '16px', 'opacity': '0.9', 'textAlign': 'center'})
        ], style={'padding': '20px'})
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': 'white', 'marginBottom': '30px',
        'borderRadius': '0 0 20px 20px'
    }),
    
    # Enhanced Tabs with icons
    dcc.Tabs(id="tabs", value='analytics', children=[
        dcc.Tab(label='Model Comparison', value='comparison', 
               style={'padding': '12px 20px'}, selected_style={'backgroundColor': colors['primary'], 'color': 'white'}),
        dcc.Tab(label='Price Prediction', value='prediction',
               style={'padding': '12px 20px'}, selected_style={'backgroundColor': colors['primary'], 'color': 'white'}),
        dcc.Tab(label='Data Analytics', value='analytics',
               style={'padding': '12px 20px'}, selected_style={'backgroundColor': colors['primary'], 'color': 'white'}),
        dcc.Tab(label='Market Insights', value='insights',
               style={'padding': '12px 20px'}, selected_style={'backgroundColor': colors['primary'], 'color': 'white'})
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='tab-content', className='tab-content', style={'padding': '0 30px', 'maxWidth': '1200px', 'margin': '0 auto'})
], style={
    'fontFamily': '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif', 
    'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'margin': '0', 'padding': '0'
})

@callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_content(active_tab):
    if active_tab == 'comparison':
        return html.Div([
            # Key Metrics Cards - Horizontal alignment fixed
            html.Div([
                html.Div([
                    html.H3("3", style={'fontSize': '36px', 'margin': '0'}),
                    html.P("ML Models", style={'margin': '5px 0'})
                ], className='metric-card', style={
                    'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%', 
                    'verticalAlign': 'top', 'minHeight': '120px'
                }),
                
                html.Div([
                    html.H3("91.01%", style={'fontSize': '36px', 'margin': '0'}),
                    html.P("Best R² Score", style={'margin': '5px 0'})
                ], className='metric-card', style={
                    'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%', 
                    'verticalAlign': 'top', 'minHeight': '120px'
                }),
                
                html.Div([
                    html.H3("74.05%", style={'fontSize': '36px', 'margin': '0'}),
                    html.P("Classification Accuracy", style={'margin': '5px 0'})
                ], className='metric-card', style={
                    'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%', 
                    'verticalAlign': 'top', 'minHeight': '120px'
                })
            ], style={'textAlign': 'center', 'marginBottom': '40px', 'whiteSpace': 'nowrap'}),
            
            html.Div([
                html.H2("Assignment Evolution & Results", style={'color': colors['dark'], 'marginBottom': '30px', 'textAlign': 'center'}),
                
                # Model comparison table with better spacing
                html.Div([
                    dash_table.DataTable(
                        data=model_comparison.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in model_comparison.columns],
                        style_cell={
                            'textAlign': 'left', 'padding': '15px', 'fontSize': '14px',
                            'fontFamily': 'inherit', 'border': '1px solid #e1e8ed',
                            'whiteSpace': 'normal', 'height': 'auto'
                        },
                        style_header={
                            'backgroundColor': colors['primary'], 'color': 'white', 
                            'fontWeight': 'bold', 'fontSize': '15px', 'padding': '15px'
                        },
                        style_data_conditional=[{
                            'if': {'row_index': 2}, 
                            'backgroundColor': '#e8f5e8', 'border': '2px solid #27ae60'
                        }],
                        style_table={'overflowX': 'auto', 'marginBottom': '30px'},
                        style_data={'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '20px'}
                    )
                ], style=card_style),
                
                # Chart with better spacing
                html.Div([
                    dcc.Graph(figure=create_comparison_chart(), className='chart-container')
                ], style={**card_style, 'marginTop': '30px'})
            ], style=card_style)
        ])
    
    elif active_tab == 'prediction':
        return html.Div([
            html.Div([
                html.H2("Intelligent Car Price Prediction", style={'color': colors['dark'], 'marginBottom': '20px'}),
                
                html.Div([
                    # Input form - better spacing and alignment
                    html.Div([
                        html.H4("Vehicle Configuration", style={'color': colors['primary'], 'textAlign': 'center', 'marginBottom': '20px'}),
                        
                        html.Label("Select ML Model:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[
                                {'label': 'A1 - Linear Regression (R² = 76.57%)', 'value': 'A1'},
                                {'label': 'A2 - Enhanced Regression (R² = 91.01%)', 'value': 'A2'},
                                {'label': 'A3 - Smart Classification (Acc = 74.05%)', 'value': 'A3'}
                            ],
                            value='A3',
                            style={'marginBottom': '20px'}
                        ),
                        
                        html.Div([
                            html.Div([
                                html.Label("Year:", style={'fontWeight': 'bold'}),
                                dcc.Input(id='year-input', type='number', value=2015, 
                                         min=2000, max=2024, style=input_style)
                            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                            
                            html.Div([
                                html.Label("KM Driven:", style={'fontWeight': 'bold'}),
                                dcc.Input(id='km-input', type='number', value=50000, 
                                         min=0, style=input_style)
                            ], style={'width': '48%', 'display': 'inline-block'})
                        ], style={'marginBottom': '15px'}),
                        
                        html.Div([
                            html.Div([
                                html.Label("Fuel Type:", style={'fontWeight': 'bold'}),
                                dcc.Dropdown(id='fuel-dropdown',
                                           options=[{'label': 'Petrol', 'value': 'Petrol'},
                                                  {'label': 'Diesel', 'value': 'Diesel'},
                                                  {'label': 'CNG', 'value': 'CNG'}],
                                           value='Petrol', style={'marginBottom': '20px'})
                            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                            
                            html.Div([
                                html.Label("Seller Type:", style={'fontWeight': 'bold'}),
                                dcc.Dropdown(id='seller-dropdown',
                                           options=[{'label': 'Individual', 'value': 'Individual'},
                                                  {'label': 'Dealer', 'value': 'Dealer'}],
                                           value='Individual', style={'marginBottom': '20px'})
                            ], style={'width': '48%', 'display': 'inline-block'})
                        ], style={'marginBottom': '15px'}),
                        
                        html.Div([
                            html.Div([
                                html.Label("Transmission:", style={'fontWeight': 'bold'}),
                                dcc.Dropdown(id='transmission-dropdown',
                                           options=[{'label': 'Manual', 'value': 'Manual'},
                                                  {'label': 'Automatic', 'value': 'Automatic'}],
                                           value='Manual', style={'marginBottom': '20px'})
                            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                            
                            html.Div([
                                html.Label("Owner Type:", style={'fontWeight': 'bold'}),
                                dcc.Dropdown(id='owner-dropdown',
                                           options=[{'label': 'First Owner', 'value': 'First Owner'},
                                                  {'label': 'Second Owner', 'value': 'Second Owner'},
                                                  {'label': 'Third Owner', 'value': 'Third Owner'},
                                                  {'label': 'Fourth & Above', 'value': 'Fourth & Above Owner'}],
                                           value='First Owner', style={'marginBottom': '20px'})
                            ], style={'width': '48%', 'display': 'inline-block'})
                        ], style={'marginBottom': '15px'}),
                        
                        html.Div([
                            html.Div([
                                html.Label("Mileage (kmpl):", style={'fontWeight': 'bold'}),
                                dcc.Input(id='mileage-input', type='number', value=20, 
                                         min=5, max=50, style=input_style)
                            ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                            
                            html.Div([
                                html.Label("Engine (CC):", style={'fontWeight': 'bold'}),
                                dcc.Input(id='engine-input', type='number', value=1200, 
                                         min=500, max=5000, style=input_style)
                            ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                            
                            html.Div([
                                html.Label("Max Power (bhp):", style={'fontWeight': 'bold'}),
                                dcc.Input(id='power-input', type='number', value=80, 
                                         min=30, max=500, style=input_style)
                            ], style={'width': '32%', 'display': 'inline-block'})
                        ], style={'marginBottom': '15px'}),
                        
                        html.Label("Seats:", style={'fontWeight': 'bold'}),
                        dcc.Input(id='seats-input', type='number', value=5, 
                                 min=2, max=8, style=input_style),
                        
                        html.Div(style={'height': '20px'}),
                        html.Button('PREDICT PRICE', id='predict-button', n_clicks=0, 
                                   style={
                                       'backgroundColor': '#3498db', 'color': 'white', 
                                       'padding': '15px 30px', 'border': 'none', 
                                       'borderRadius': '8px', 'cursor': 'pointer', 
                                       'width': '100%', 'fontSize': '16px', 
                                       'fontWeight': 'bold', 'marginTop': '15px',
                                       'textTransform': 'uppercase', 'letterSpacing': '1px',
                                       'display': 'flex', 'alignItems': 'center', 
                                       'justifyContent': 'center', 'minHeight': '50px'
                                   })
                        
                    ], style={
                        'width': '47%', 'display': 'inline-block', 'verticalAlign': 'top',
                        'margin': '0 1.5%', 'backgroundColor': 'white', 'padding': '20px',
                        'borderRadius': '12px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
                    }),
                    
                    # Results - better spacing and alignment
                    html.Div([
                        html.H4("Prediction Results", style={'color': colors['primary'], 'textAlign': 'center', 'marginBottom': '20px'}),
                        html.Div(id='prediction-output', style={
                            'padding': '20px', 'backgroundColor': '#f8f9fa',
                            'borderRadius': '12px', 'minHeight': '300px',
                            'border': '2px dashed #dee2e6'
                        })
                    ], style={
                        'width': '47%', 'display': 'inline-block', 'verticalAlign': 'top',
                        'margin': '0 1.5%', 'backgroundColor': 'white', 'padding': '20px',
                        'borderRadius': '12px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
                    })
                    
                ], style={'textAlign': 'center'})
            ], style=card_style)
        ])
    
    elif active_tab == 'analytics':
        return html.Div([
            html.H2("Advanced Data Analytics", style={'color': colors['dark'], 'marginBottom': '30px'}),
            
            # Summary Statistics
            html.Div([
                html.H3("Dataset Overview", style={'marginBottom': '20px'}),
                html.Div(id='dataset-stats')
            ], style=card_style),
            
            # Charts Grid - 2 per row with proper spacing
            html.Div([
                html.Div([
                    dcc.Graph(id='price-dist', className='chart-container')
                ], style={
                    'width': '47%', 'display': 'inline-block', 'margin': '0 1.5%',
                    'verticalAlign': 'top', 'backgroundColor': 'white', 'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'padding': '15px'
                }),
                
                html.Div([
                    dcc.Graph(id='year-trend', className='chart-container')
                ], style={
                    'width': '47%', 'display': 'inline-block', 'margin': '0 1.5%',
                    'verticalAlign': 'top', 'backgroundColor': 'white', 'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'padding': '15px'
                })
            ], style={'marginBottom': '30px', 'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='fuel-analysis', className='chart-container')
                ], style={
                    'width': '47%', 'display': 'inline-block', 'margin': '0 1.5%',
                    'verticalAlign': 'top', 'backgroundColor': 'white', 'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'padding': '15px'
                }),
                
                html.Div([
                    dcc.Graph(id='correlation-heatmap', className='chart-container')
                ], style={
                    'width': '47%', 'display': 'inline-block', 'margin': '0 1.5%',
                    'verticalAlign': 'top', 'backgroundColor': 'white', 'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'padding': '15px'
                })
            ], style={'textAlign': 'center'})
        ])
    
    elif active_tab == 'insights':
        return html.Div([
            html.H2("Market Insights & Trends", style={'color': colors['dark'], 'marginBottom': '30px'}),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='brand-analysis', className='chart-container')
                ], style={
                    'width': '47%', 'display': 'inline-block', 'margin': '0 1.5%',
                    'verticalAlign': 'top', 'backgroundColor': 'white', 'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'padding': '15px'
                }),
                
                html.Div([
                    dcc.Graph(id='owner-impact', className='chart-container')
                ], style={
                    'width': '47%', 'display': 'inline-block', 'margin': '0 1.5%',
                    'verticalAlign': 'top', 'backgroundColor': 'white', 'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'padding': '15px'
                })
            ], style={'marginBottom': '30px', 'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='mileage-price', className='chart-container')
                ], style={
                    'width': '47%', 'display': 'inline-block', 'margin': '0 1.5%',
                    'verticalAlign': 'top', 'backgroundColor': 'white', 'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'padding': '15px'
                }),
                
                html.Div([
                    dcc.Graph(id='age-depreciation', className='chart-container')
                ], style={
                    'width': '47%', 'display': 'inline-block', 'margin': '0 1.5%',
                    'verticalAlign': 'top', 'backgroundColor': 'white', 'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'padding': '15px'
                })
            ], style={'textAlign': 'center'})
        ])

def create_comparison_chart():
    models_list = ['A1 Linear', 'A2 Enhanced', 'A3 Classification']
    scores = [0.7657, 0.9101, 0.7405]  # Updated with current actual results
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models_list, y=scores,
        marker_color=['#e74c3c', '#f39c12', '#27ae60'],
        text=[f'R²: {scores[0]:.4f}', f'R²: {scores[1]:.4f}', f'Acc: {scores[2]:.4f}'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Final Model Performance Results',
        xaxis_title='Model', yaxis_title='Score',
        template='plotly_white',
        yaxis=dict(range=[0, 1])
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
def predict_price(n_clicks, model_choice, year, km, fuel, seller, transmission, owner, mileage, engine, power, seats):
    if n_clicks == 0:
        return html.Div([
            html.P("Select model and enter car details, then click 'Predict Price'"),
            html.P("Available models:"),
            html.Ul([
                html.Li("A1: Linear Regression (R² = 0.6040)"),
                html.Li("A2: Enhanced with Polynomial Features (R² = 0.8472)"),
                html.Li("A3: Car Price Classification (74.05% accuracy)")
            ])
        ])
    
    try:
        if model_choice == 'A1' and 'A1' in models:
            model_data = models['A1']
            
            # Encode categorical variables
            fuel_encoded = model_data['label_encoders']['fuel'].transform([fuel])[0]
            seller_encoded = model_data['label_encoders']['seller_type'].transform([seller])[0]
            transmission_encoded = model_data['label_encoders']['transmission'].transform([transmission])[0]
            owner_encoded = model_data['label_encoders']['owner'].transform([owner])[0]
            
            # Create feature array (10 features: 6 numeric + 4 categorical)
            features = np.array([[year, km, mileage, engine, power, seats, fuel_encoded, seller_encoded, transmission_encoded, owner_encoded]])
            
            # Scale features
            features_scaled = model_data['scaler'].transform(features)
            
            # Predict
            prediction = model_data['model'].predict(features_scaled)[0]
            prediction = max(30000, prediction)  # Ensure positive price
            
            return html.Div([
                html.H4("A1 Linear Regression Result", style={'color': '#e74c3c'}),
                html.H3(f"Predicted Price: {prediction:,.0f}", style={'color': '#27ae60'}),
                html.P(f"Model R² Score: {model_data['metrics']['test_r2']:.4f}"),
                html.P("✅ Basic linear regression with proper pipeline"),
                html.Hr(),
                html.P("Input Summary:", style={'fontWeight': 'bold'}),
                html.P(f"Year: {year}, KM: {km:,}, Mileage: {mileage} kmpl"),
                html.P(f"Engine: {engine} CC, Power: {power} bhp, Seats: {seats}"),
                html.P(f"Fuel: {fuel}, Seller: {seller}"),
                html.P(f"Transmission: {transmission}, Owner: {owner}")
            ])
            
        elif model_choice == 'A2' and 'A2' in models:
            model_data = models['A2']
            
            # Encode categorical variables
            fuel_encoded = model_data['label_encoders']['fuel'].transform([fuel])[0]
            seller_encoded = model_data['label_encoders']['seller_type'].transform([seller])[0]
            transmission_encoded = model_data['label_encoders']['transmission'].transform([transmission])[0]
            owner_encoded = model_data['label_encoders']['owner'].transform([owner])[0]
            
            # Create feature array (10 features: 6 numeric + 4 categorical)
            features = np.array([[year, km, mileage, engine, power, seats, fuel_encoded, seller_encoded, transmission_encoded, owner_encoded]])
            
            # Scale and create polynomial features
            features_scaled = model_data['scaler'].transform(features)
            features_poly = model_data['poly'].transform(features_scaled)
            
            # Predict
            prediction = model_data['model'].predict(features_poly)[0]
            prediction = max(30000, abs(prediction))  # Ensure positive price
            
            return html.Div([
                html.H4("A2 Enhanced Linear Regression Result", style={'color': '#f39c12'}),
                html.H3(f"Predicted Price: {prediction:,.0f}", style={'color': '#27ae60'}),
                html.P(f"Model R² Score: {model_data['metrics']['test_r2']:.4f}"),
                html.P("✅ Enhanced with polynomial features and Lasso regularization"),
                html.Hr(),
                html.P("Input Summary:", style={'fontWeight': 'bold'}),
                html.P(f"Year: {year}, KM: {km:,}, Mileage: {mileage} kmpl"),
                html.P(f"Engine: {engine} CC, Power: {power} bhp, Seats: {seats}"),
                html.P(f"Fuel: {fuel}, Seller: {seller}"),
                html.P(f"Transmission: {transmission}, Owner: {owner}")
            ])
            
        elif model_choice == 'A3' and 'A3' in models:
            model_data = models['A3']
            
            # Encode categorical variables using the saved label encoders
            fuel_encoded = model_data['label_encoders']['fuel'].transform([fuel])[0]
            seller_encoded = model_data['label_encoders']['seller_type'].transform([seller])[0]
            transmission_encoded = model_data['label_encoders']['transmission'].transform([transmission])[0]
            owner_encoded = model_data['label_encoders']['owner'].transform([owner])[0]
            
            # Create feature array (10 features: 6 numeric + 4 categorical)
            features = np.array([[year, km, mileage, engine, power, seats, fuel_encoded, seller_encoded, transmission_encoded, owner_encoded]])
            
            # Scale features
            features_scaled = model_data['scaler'].transform(features)
            
            # Predict
            prediction = model_data['model'].predict(features_scaled)[0]
            
            # Price class mapping (updated to match improved model)
            price_classes = {
                0: "Low (0 - 3.5 Lakhs)",
                1: "Medium (3.5 - 7 Lakhs)", 
                2: "High (7 - 15 Lakhs)",
                3: "Premium (Above 15 Lakhs)"
            }
            
            class_name = price_classes.get(int(prediction), "Unknown")
            
            return html.Div([
                html.H4("A3 Logistic Classification Result", style={'color': '#27ae60'}),
                html.H3(f"Price Class {int(prediction)}: {class_name}", style={'color': '#3498db'}),
                html.P("Model Accuracy: 74.05%", style={'fontSize': '16px', 'fontWeight': 'bold'}),
                html.P("Macro F1-Score: 0.6842"),
                html.P("✅ Custom logistic regression with Ridge regularization"),
                html.Hr(),
                html.Div([
                    html.P("💡 To see different price classes, try:", style={'fontWeight': 'bold', 'color': '#2980b9'}),
                    html.P("• Class 0 (Low): Old cars (2005-2010), high km (>100k), basic specs"),
                    html.P("• Class 1 (Medium): Mid cars (2012-2016), moderate km (50-100k), standard specs"),
                    html.P("• Class 2 (High): Recent cars (2017-2020), low km (<50k), large engine (>1800cc), 7 seats"),
                    html.P("• Class 3 (Premium): New cars (2020+), very low km (<20k), high power (>200bhp), luxury features")
                ], style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px', 'margin': '10px 0'}),
                html.Hr(),
                html.P("Input Summary:", style={'fontWeight': 'bold'}),
                html.P(f"Year: {year}, KM: {km:,}, Mileage: {mileage} kmpl"),
                html.P(f"Engine: {engine} CC, Power: {power} bhp, Seats: {seats}"),
                html.P(f"Fuel: {fuel}, Seller: {seller}"),
                html.P(f"Transmission: {transmission}, Owner: {owner}")
            ])
        else:
            return html.Div([
                html.H4("Model Not Available", style={'color': '#e74c3c'}),
                html.P(f"Model {model_choice} is not loaded."),
                html.P("Available models: " + ", ".join(models.keys()))
            ])
            
    except Exception as e:
        return html.Div([
            html.H4("Prediction Error", style={'color': '#e74c3c'}),
            html.P(f"Error: {str(e)}"),
            html.P("Please check your inputs and try again."),
            html.P("Make sure all fields are filled correctly.")
        ])

# Enhanced analytics callbacks
@callback(Output('dataset-stats', 'children'), Input('tabs', 'value'))
def update_dataset_stats(tab):
    if data.empty:
        return html.P("No data available")
    
    stats = [
        html.Div([
            html.H4(f"{len(data):,}", style={'margin': '0', 'color': colors['primary']}),
            html.P("Total Cars", style={'margin': '0'})
        ], className='metric-card', style={'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.H4(f"{data['selling_price'].mean():,.0f}", style={'margin': '0', 'color': colors['secondary']}),
            html.P("Avg Price", style={'margin': '0'})
        ], className='metric-card', style={'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.H4(f"{data['year'].min()}-{data['year'].max()}", style={'margin': '0', 'color': colors['warning']}),
            html.P("Year Range", style={'margin': '0'})
        ], className='metric-card', style={'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.H4(f"{data['fuel'].nunique()}", style={'margin': '0', 'color': colors['accent']}),
            html.P("Fuel Types", style={'margin': '0'})
        ], className='metric-card', style={'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.H4(f"{data['km_driven'].median():,.0f}", style={'margin': '0', 'color': colors['dark']}),
            html.P("Median KM", style={'margin': '0'})
        ], className='metric-card', style={'width': '18%', 'display': 'inline-block', 'margin': '1%'})
    ]
    
    return html.Div(stats, style={'textAlign': 'center'})

@callback(Output('price-dist', 'figure'), Input('tabs', 'value'))
def update_price_dist(tab):
    if data.empty:
        return go.Figure()
    
    fig = px.histogram(data, x='selling_price', nbins=40, 
                      title='Price Distribution Analysis',
                      labels={'selling_price': 'Selling Price', 'count': 'Number of Cars'},
                      color_discrete_sequence=[colors['primary']])
    
    fig.add_vline(x=data['selling_price'].mean(), line_dash="dash", 
                  annotation_text=f"Mean: {data['selling_price'].mean():,.0f}",
                  annotation_position="top")
    fig.add_vline(x=data['selling_price'].median(), line_dash="dot", 
                  annotation_text=f"Median: {data['selling_price'].median():,.0f}",
                  annotation_position="bottom")
    
    fig.update_layout(template='plotly_white', showlegend=False,
                     title_font_size=16, title_x=0.5)
    return fig

@callback(Output('year-trend', 'figure'), Input('tabs', 'value'))
def update_year_trend(tab):
    if data.empty:
        return go.Figure()
    
    yearly_stats = data.groupby('year').agg({
        'selling_price': ['mean', 'count', 'std']
    }).round(0)
    yearly_stats.columns = ['avg_price', 'count', 'std_price']
    yearly_stats = yearly_stats.reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=yearly_stats['year'], y=yearly_stats['avg_price'],
                  mode='lines+markers', name='Average Price',
                  line=dict(color=colors['primary'], width=3),
                  marker=dict(size=8)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=yearly_stats['year'], y=yearly_stats['count'],
               name='Car Count', opacity=0.3,
               marker_color=colors['secondary']),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Average Price", secondary_y=False)
    fig.update_yaxes(title_text="Number of Cars", secondary_y=True)
    fig.update_layout(title='Price Trends Over Years', template='plotly_white',
                     title_font_size=16, title_x=0.5)
    return fig

@callback(Output('fuel-analysis', 'figure'), Input('tabs', 'value'))
def update_fuel_analysis(tab):
    if data.empty:
        return go.Figure()
    
    fuel_stats = data.groupby('fuel')['selling_price'].agg(['mean', 'count']).reset_index()
    
    fig = px.box(data, x='fuel', y='selling_price', 
                title='Price Distribution by Fuel Type',
                labels={'selling_price': 'Price', 'fuel': 'Fuel Type'},
                color='fuel', color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_layout(template='plotly_white', showlegend=False,
                     title_font_size=16, title_x=0.5)
    return fig

@callback(Output('correlation-heatmap', 'figure'), Input('tabs', 'value'))
def update_correlation_heatmap(tab):
    if data.empty:
        return go.Figure()
    
    try:
        # Clean and prepare numeric columns
        data_clean = data.copy()
        
        # Clean mileage column (remove 'kmpl')
        if 'mileage' in data_clean.columns:
            data_clean['mileage'] = pd.to_numeric(data_clean['mileage'].astype(str).str.replace(' kmpl', '').str.replace(' km/kg', ''), errors='coerce')
        
        # Clean engine column (remove 'CC')
        if 'engine' in data_clean.columns:
            data_clean['engine'] = pd.to_numeric(data_clean['engine'].astype(str).str.replace(' CC', ''), errors='coerce')
        
        # Clean max_power column (remove 'bhp')
        if 'max_power' in data_clean.columns:
            data_clean['max_power'] = pd.to_numeric(data_clean['max_power'].astype(str).str.replace(' bhp', ''), errors='coerce')
        
        # Select numeric columns for correlation
        numeric_cols = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        available_cols = [col for col in numeric_cols if col in data_clean.columns]
        
        if len(available_cols) < 2:
            return go.Figure().add_annotation(text="Not enough numeric data for correlation", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Calculate correlation matrix
        corr_data = data_clean[available_cols].select_dtypes(include=[np.number])
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(corr_matrix, 
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu_r',
                       aspect='auto',
                       text_auto=True)
        
        fig.update_layout(template='plotly_white', title_font_size=16, title_x=0.5)
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(text=f"Error loading correlation data", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

@callback(Output('brand-analysis', 'figure'), Input('tabs', 'value'))
def update_brand_analysis(tab):
    if data.empty or 'name' not in data.columns:
        return go.Figure()
    
    # Extract brand from car name
    data['brand'] = data['name'].str.split().str[0]
    brand_stats = data.groupby('brand').agg({
        'selling_price': 'mean',
        'name': 'count'
    }).round(0)
    brand_stats.columns = ['avg_price', 'count']
    brand_stats = brand_stats[brand_stats['count'] >= 10].sort_values('avg_price', ascending=True)
    
    fig = px.bar(brand_stats.reset_index(), x='avg_price', y='brand',
                title='Average Price by Brand (Min 10 cars)',
                labels={'avg_price': 'Average Price', 'brand': 'Brand'},
                orientation='h', color='avg_price',
                color_continuous_scale='viridis')
    
    fig.update_layout(template='plotly_white', title_font_size=16, title_x=0.5)
    return fig

@callback(Output('owner-impact', 'figure'), Input('tabs', 'value'))
def update_owner_impact(tab):
    if data.empty:
        return go.Figure()
    
    owner_stats = data.groupby('owner')['selling_price'].agg(['mean', 'count']).reset_index()
    
    fig = px.bar(owner_stats, x='owner', y='mean',
                title='Price Impact by Owner Type',
                labels={'mean': 'Average Price', 'owner': 'Owner Type'},
                color='mean', color_continuous_scale='blues')
    
    fig.update_layout(template='plotly_white', title_font_size=16, title_x=0.5)
    return fig

@callback(Output('mileage-price', 'figure'), Input('tabs', 'value'))
def update_mileage_price(tab):
    if data.empty:
        return go.Figure()
    
    try:
        # Clean data
        data_clean = data.copy()
        
        # Clean mileage column
        if 'mileage' in data_clean.columns:
            data_clean['mileage'] = pd.to_numeric(data_clean['mileage'].astype(str).str.replace(' kmpl', '').str.replace(' km/kg', ''), errors='coerce')
        
        # Clean engine column for size
        if 'engine' in data_clean.columns:
            data_clean['engine'] = pd.to_numeric(data_clean['engine'].astype(str).str.replace(' CC', ''), errors='coerce')
        
        # Remove rows with missing mileage data
        data_clean = data_clean.dropna(subset=['mileage', 'selling_price'])
        
        if data_clean.empty:
            return go.Figure().add_annotation(text="No valid mileage data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Create scatter plot
        fig = px.scatter(data_clean, x='mileage', y='selling_price', 
                        color='fuel' if 'fuel' in data_clean.columns else None,
                        size='engine' if 'engine' in data_clean.columns and not data_clean['engine'].isna().all() else None,
                        title='Mileage vs Price Analysis',
                        labels={'mileage': 'Mileage (kmpl)', 'selling_price': 'Price'},
                        hover_data=['year', 'km_driven'] if all(col in data_clean.columns for col in ['year', 'km_driven']) else None)
        
        fig.update_layout(template='plotly_white', title_font_size=16, title_x=0.5)
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(text="Error loading mileage data", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

@callback(Output('age-depreciation', 'figure'), Input('tabs', 'value'))
def update_age_depreciation(tab):
    if data.empty:
        return go.Figure()
    
    current_year = 2024
    data['age'] = current_year - data['year']
    
    age_stats = data.groupby('age')['selling_price'].mean().reset_index()
    
    fig = px.line(age_stats, x='age', y='selling_price',
                 title='Car Depreciation by Age',
                 labels={'age': 'Car Age (Years)', 'selling_price': 'Average Price'},
                 markers=True, line_shape='spline')
    
    fig.update_traces(line=dict(color=colors['accent'], width=3),
                     marker=dict(size=8, color=colors['accent']))
    fig.update_layout(template='plotly_white', title_font_size=16, title_x=0.5)
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
