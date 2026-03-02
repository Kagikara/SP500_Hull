"""
Hull Tactical Market Prediction - Simplified XGBoost Version
Optimized for Kaggle competition constraints
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.default_inference_server


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function with numerical stability"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def platt_scaling_fit(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Fit Platt Scaling parameters using simple optimization
    P(y=1|score) = sigmoid(A*score + B)
    """
    from scipy.optimize import minimize
    
    def neg_log_likelihood(params):
        A, B = params
        z = A * scores + B
        probs = sigmoid(z)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
    
    result = minimize(neg_log_likelihood, [1.0, 0.0], method='BFGS')
    return {'A': result.x[0], 'B': result.x[1]}


def platt_scaling_transform(scores: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Apply Platt Scaling transformation"""
    z = params['A'] * scores + params['B']
    return sigmoid(z)


# ============================================================
# DATA PROCESSOR
# ============================================================

class DataProcessor:
    """Efficient data processor"""
    
    def __init__(self, missing_threshold: float = 0.20):
        self.missing_threshold = missing_threshold
        self.feature_cols = None
        self.feature_means = None
        
    def process_training_data(self, data_path: str):
        """Process training data"""
        print("[Data] Loading training data...")
        
        # Read data
        df = pd.read_csv(data_path)
        
        # Filter by date
        df = df[df['date_id'] >= 1006].sort_values('date_id').reset_index(drop=True)
        
        # Identify features
        target_col = "market_forward_excess_returns"
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', target_col]
        potential_features = [col for col in df.columns if col not in exclude_cols]
        
        # Remove high missing rate features
        missing_ratio = df[potential_features].isnull().mean()
        valid_features = missing_ratio[missing_ratio <= self.missing_threshold].index.tolist()
        
        print(f"  - Selected {len(valid_features)} features")
        
        # Fill missing values
        df[valid_features] = df[valid_features].fillna(method='ffill').fillna(0)
        
        # Remove rows with missing target
        df = df[df[target_col].notna()].copy()
        
        # Create binary target
        df['y'] = (df[target_col] >= 0).astype(int)
        
        # Split train/test
        train_df = df[df['date_id'] <= 7000].copy()
        test_df = df[(df['date_id'] >= 7001) & (df['date_id'] <= 7984)].copy()
        
        # Remove warmup period
        min_train_date = train_df['date_id'].min() + 250
        train_df = train_df[train_df['date_id'] > min_train_date].copy()
        
        print(f"  - Training samples: {len(train_df)}")
        print(f"  - Test samples: {len(test_df)}")
        
        # Store feature info
        self.feature_cols = valid_features
        self.feature_means = train_df[valid_features].mean().to_dict()
        
        return {
            'X_train': train_df[valid_features].values,
            'y_train': train_df['y'].values,
            'X_test': test_df[valid_features].values,
            'y_test': test_df['y'].values,
            'train_df': train_df,
            'test_df': test_df
        }
    
    def process_prediction_data(self, test_df: pl.DataFrame):
        """Process incoming prediction data"""
        # Convert to pandas
        df_pd = test_df.to_pandas()
        
        # Ensure all features exist
        for feat in self.feature_cols:
            if feat not in df_pd.columns:
                df_pd[feat] = self.feature_means.get(feat, 0)
        
        # Fill missing
        df_pd[self.feature_cols] = df_pd[self.feature_cols].fillna(method='ffill')
        for feat in self.feature_cols:
            df_pd[feat] = df_pd[feat].fillna(self.feature_means.get(feat, 0))
        
        return df_pd[self.feature_cols].values


# ============================================================
# XGBOOST MODEL
# ============================================================

class XGBoostPredictor:
    """XGBoost-based predictor with calibration"""
    
    def __init__(self):
        self.model = None
        self.platt_params = None
        self.data_processor = DataProcessor()
        
    def train(self, data_path: str):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        # Process data
        data = self.data_processor.process_training_data(data_path)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Import XGBoost
        import xgboost as xgb
        
        # Split for validation
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
        
        print("\n[XGBoost] Training model...")
        
        # Configure parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.01,
            'max_depth': 1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'random_state': 42,
            'tree_method': 'hist',  # Faster training
            'n_jobs': -1
        }
        
        # Train model
        self.model = xgb.XGBClassifier(
            n_estimators=1000,
            early_stopping_rounds=50,
            **params
        )
        
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        best_iteration = self.model.best_iteration
        print(f"  - Best iteration: {best_iteration}")
        
        # Calibrate on training set
        print("\n[Calibration] Fitting Platt Scaling...")
        
        # Get training probabilities
        train_probs = self.model.predict_proba(X_train)[:, 1]
        
        # Convert to logit scores
        epsilon = 1e-6
        train_probs_clipped = np.clip(train_probs, epsilon, 1 - epsilon)
        train_scores = np.log(train_probs_clipped / (1 - train_probs_clipped))
        train_scores = np.clip(train_scores, -20, 20)
        
        # Fit Platt parameters
        self.platt_params = platt_scaling_fit(train_scores, y_train)
        
        print(f"  - Platt A: {self.platt_params['A']:.4f}")
        print(f"  - Platt B: {self.platt_params['B']:.4f}")
        
        # Evaluate on test set
        print("\n[Evaluation] Testing model...")
        test_probs = self.model.predict_proba(X_test)[:, 1]
        test_probs_clipped = np.clip(test_probs, epsilon, 1 - epsilon)
        test_scores = np.log(test_probs_clipped / (1 - test_probs_clipped))
        test_scores = np.clip(test_scores, -20, 20)
        
        # Apply calibration
        calibrated_probs = platt_scaling_transform(test_scores, self.platt_params)
        
        # Simple evaluation
        predictions = (calibrated_probs > 0.5).astype(int)
        accuracy = np.mean(predictions == y_test)
        
        print(f"  - Test accuracy: {accuracy:.4f}")
        print(f"  - Calibrated prob range: [{calibrated_probs.min():.4f}, {calibrated_probs.max():.4f}]")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
    def predict(self, test_df: pl.DataFrame) -> float:
        """Predict position for test data"""
        
        # Process data
        X_pred = self.data_processor.process_prediction_data(test_df)
        
        # Get probabilities
        probs = self.model.predict_proba(X_pred)[:, 1]
        
        # Convert to logit scores
        epsilon = 1e-6
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        scores = np.log(probs_clipped / (1 - probs_clipped))
        scores = np.clip(scores, -20, 20)
        
        # Apply calibration
        calibrated_probs = platt_scaling_transform(scores, self.platt_params)
        
        # Get average probability for the batch
        avg_prob = float(np.mean(calibrated_probs))
        
        # Convert to position
        # Use a simple threshold: prob > 0.5 → long, else no position
        position = 1.0 if avg_prob > 0.5 else 0.0
        
        return position


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

# Global predictor
predictor = None
initialized = False


def predict(test: pl.DataFrame) -> float:
    """
    Main prediction function for Kaggle API
    
    Args:
        test: Polars DataFrame with test features
        
    Returns:
        float: Position (0.0 or 1.0)
    """
    global predictor, initialized
    
    # Initialize on first call
    if not initialized:
        print("\n" + "="*60)
        print("INITIALIZING PREDICTOR")
        print("="*60)
        
        try:
            # Create predictor
            predictor = XGBoostPredictor()
            
            # Train on available data
            train_path = '/kaggle/input/hull-tactical-market-prediction/train.csv'
            
            if os.path.exists(train_path):
                predictor.train(train_path)
                initialized = True
                print("\nInitialization successful!\n")
            else:
                print(f"Warning: Training file not found at {train_path}")
                return 0.0
                
        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    # Make prediction
    try:
        position = predictor.predict(test)
        return position
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


# ============================================================
# KAGGLE API INTEGRATION
# ============================================================

inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
