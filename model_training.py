import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.onehot_encoder = None
        self.onehot_columns = None
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the housing data"""
        # Load the data
        df = pd.read_csv(csv_path)
        
        print("Dataset Overview:")
        print(f"Shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Handle column name inconsistencies
        df.columns = df.columns.str.strip()
        
        # Check for missing values
        if df.isnull().sum().any():
            print("Handling missing values...")
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def feature_engineering(self, df):
        """Perform feature engineering"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Feature engineering: Create new features
        data['House_Age'] = 2024 - data['Year_Built']
        data['Rooms_Total'] = data['Num_Bedrooms'] + data['Num_Bathrooms']
        
        # Remove Price_Per_Sqft since it requires target variable during prediction
        # data['Price_Per_Sqft'] = data['Price'] / data['Square_Feet']
        
        # Remove the original Year_Built to avoid multicollinearity
        data = data.drop('Year_Built', axis=1)
        
        return data
    
    def prepare_features(self, data):
        """Prepare features for modeling"""
        # Separate features and target
        X = data.drop('Price', axis=1)
        y = data['Price']
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
        # One-hot encode categorical variables
        if categorical_cols:
            self.onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
            X_encoded = self.onehot_encoder.fit_transform(X[categorical_cols])
            self.onehot_columns = self.onehot_encoder.get_feature_names_out(categorical_cols)
            
            # Combine with numerical features
            X_numerical = X[numerical_cols].values
            X_processed = np.column_stack([X_numerical, X_encoded])
            feature_names = numerical_cols + self.onehot_columns.tolist()
        else:
            X_processed = X.values
            feature_names = numerical_cols
        
        self.feature_names = feature_names
        print(f"Total features after preprocessing: {len(feature_names)}")
        return X_processed, y, feature_names
    
    def train_models(self, X, y, test_size=0.2):
        """Train multiple models and select the best one"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_score = float('inf')
        best_model = None
        best_model_name = None
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Random Forest':
                # Hyperparameter tuning for Random Forest
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            
            elif name == 'XGBoost':
                # Hyperparameter tuning for XGBoost
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1]
                }
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            
            else:
                model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'R2 Score': r2,
                'MSE': mse,
                'model': model
            }
            
            results[name] = metrics
            
            print(f"{name} Performance:")
            print(f"RMSE: {rmse:,.2f}")
            print(f"MAE: {mae:,.2f}")
            print(f"R2 Score: {r2:.4f}")
            
            # Update best model
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_model_name = name
        
        print(f"\n🎯 Best Model: {best_model_name} with RMSE: {best_score:,.2f}")
        
        self.model = best_model
        return X_test, y_test, best_model.predict(X_test_scaled), results
    
    def create_visualizations(self, X_test, y_test, y_pred, results):
        """Create professional visualizations"""
        # 1. Prediction vs Actual scatter plot
        fig1 = px.scatter(
            x=y_test, y=y_pred,
            labels={'x': 'Actual Prices', 'y': 'Predicted Prices'},
            title='Actual vs Predicted House Prices',
            trendline='ols'
        )
        fig1.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Perfect Prediction'
        ))
        
        # 2. Residuals histogram
        residuals = y_test - y_pred
        fig2 = px.histogram(
            x=residuals,
            nbins=50,
            labels={'x': 'Prediction Error (Residuals)'},
            title='Distribution of Prediction Errors'
        )
        
        # 3. Feature importance (if using tree-based model)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig3 = px.bar(
                feature_importance.tail(20),  # Show top 20 features
                x='importance',
                y='feature',
                orientation='h',
                title='Top 20 Feature Importance in Price Prediction'
            )
        else:
            fig3 = None
        
        # 4. Model comparison
        model_names = list(results.keys())
        rmse_scores = [results[name]['RMSE'] for name in model_names]
        
        fig4 = px.bar(
            x=model_names, y=rmse_scores,
            labels={'x': 'Models', 'y': 'RMSE'},
            title='Model Comparison (Lower RMSE is Better)',
            color=rmse_scores,
            color_continuous_scale='Viridis'
        )
        
        return fig1, fig2, fig3, fig4, residuals
    
    def save_model(self, filename='house_price_model.joblib'):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'onehot_encoder': self.onehot_encoder,
            'feature_names': self.feature_names,
            'onehot_columns': self.onehot_columns
        }
        joblib.dump(model_data, filename)
        print(f"Model saved successfully as {filename}")
    
    def load_model(self, filename='house_price_model.joblib'):
        """Load the trained model and preprocessing objects"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.onehot_encoder = model_data['onehot_encoder']
        self.feature_names = model_data['feature_names']
        self.onehot_columns = model_data['onehot_columns']
        print(f"Loaded model expecting {len(self.feature_names)} features")
        return self
    
    def predict_price(self, features_dict):
        """Predict price for new house features"""
        # Convert features to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Feature engineering (must match training exactly)
        features_df['House_Age'] = 2024 - features_df['Year_Built']
        features_df['Rooms_Total'] = features_df['Num_Bedrooms'] + features_df['Num_Bathrooms']
        features_df = features_df.drop('Year_Built', axis=1)
        
        # Separate categorical and numerical features
        categorical_cols = features_df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Prediction - Categorical columns: {categorical_cols}")
        print(f"Prediction - Numerical columns: {numerical_cols}")
        print(f"Prediction - Total features: {len(numerical_cols) + (len(self.onehot_columns) if categorical_cols else 0)}")
        
        # One-hot encode categorical variables
        if categorical_cols and self.onehot_encoder is not None:
            X_encoded = self.onehot_encoder.transform(features_df[categorical_cols])
            X_numerical = features_df[numerical_cols].values
            X_processed = np.column_stack([X_numerical, X_encoded])
        else:
            X_processed = features_df.values
        
        # Scale features and predict
        features_scaled = self.scaler.transform(X_processed)
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction

# Main execution
if __name__ == "__main__":
    predictor = HousePricePredictor()
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    housing_data = predictor.load_and_preprocess_data('synthetic_house_prices.csv')
    
    # Feature engineering
    print("🔧 Performing feature engineering...")
    housing_data_engineered = predictor.feature_engineering(housing_data)
    
    # Prepare features
    print("⚙️ Preparing features for modeling...")
    X_processed, y, feature_names = predictor.prepare_features(housing_data_engineered)
    
    # Train models
    print("🤖 Training machine learning models...")
    X_test, y_test, y_pred, results = predictor.train_models(X_processed, y)
    
    # Create visualizations
    print("📈 Creating visualizations...")
    fig1, fig2, fig3, fig4, residuals = predictor.create_visualizations(X_test, y_test, y_pred, results)
    
    # Save the model
    predictor.save_model()
    
    # Sample prediction
    sample_house = {
        'Square_Feet': 2000,
        'Num_Bedrooms': 3,
        'Num_Bathrooms': 2,
        'Lot_Size_Acres': 0.5,
        'Year_Built': 1990,
        'Garage_Spaces': 2,
        'Neighborhood': 'Suburb'
    }
    
    predicted_price = predictor.predict_price(sample_house)
    print(f"\n🏠 Sample Prediction:")
    print(f"Features: {sample_house}")
    print(f"Predicted Price: ${predicted_price:,.2f}")
    
    print("\n✅ Model training completed successfully!")