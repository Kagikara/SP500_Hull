"""
Hull Tactical Market Prediction - Full Ensemble Version
Faithful translation combining XGBoost, Random Forest, and Logistic Regression
with all R code strategies preserved
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.default_inference_server


# ============================================================
# SHARED ALGORITHM LIBRARY
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function with clipping"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def algo_platt_fit(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Platt Scaling calibration"""
    from scipy.optimize import minimize
    
    def objective(params):
        A, B = params
        z = A * scores + B
        z = np.clip(z, -500, 500)
        probs = 1 / (1 + np.exp(z))
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
        return loss
    
    result = minimize(objective, [1.0, 0.0], method='BFGS')
    return {'A': result.x[0], 'B': result.x[1]}


def algo_platt_predict(scores: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Apply Platt Scaling"""
    z = params['A'] * scores + params['B']
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(z))


# ============================================================
# XGBOOST COMPONENT
# ============================================================

class XGBoostComponent:
    """XGBoost model component"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.platt_params = None
        self.feature_cols = None
    
    def train(self, file_path: str) -> bool:
        """Train XGBoost model"""
        print("\n[XGBoost] Training component...")
        
        try:
            import xgboost as xgb
            
            # Load data
            df = pd.read_csv(file_path)
            df = df[df['date_id'] >= 1006].sort_values('date_id')
            
            # Feature selection
            target_col = "market_forward_excess_returns"
            exclude = ['date_id', 'forward_returns', 'risk_free_rate', target_col]
            features = [c for c in df.columns if c not in exclude]
            
            # Missing rate filter
            missing = df[features].isnull().mean()
            self.feature_cols = missing[missing <= 0.20].index.tolist()
            
            # Prepare data
            df_clean = df[['date_id', target_col] + self.feature_cols].copy()
            df_clean = df_clean[df_clean[target_col].notna()]
            df_clean['y'] = (df_clean[target_col] >= 0).astype(int)
            
            # Split
            train = df_clean[df_clean['date_id'] <= 7000]
            
            X_train = train[self.feature_cols].values
            y_train = train['y'].values
            
            # Train/val split
            split = int(len(X_train) * 0.8)
            X_tr, X_val = X_train[:split], X_train[split:]
            y_tr, y_val = y_train[:split], y_train[split:]
            
            # Train model
            params = self.config['xgb_params']
            self.model = xgb.XGBClassifier(
                objective=params['objective'],
                learning_rate=params['eta'],
                max_depth=params['max_depth'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                n_estimators=params['nrounds'],
                early_stopping_rounds=params['early_stopping'],
                random_state=42,
                tree_method='hist',
                n_jobs=-1,
                verbosity=0
            )
            
            self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            # Fit Platt on training data
            train_probs = self.model.predict_proba(X_train)[:, 1]
            epsilon = 1e-6
            probs_clip = np.clip(train_probs, epsilon, 1 - epsilon)
            scores = np.log(probs_clip / (1 - probs_clip))
            scores = np.clip(scores, -20, 20)
            
            self.platt_params = algo_platt_fit(scores, y_train)
            
            print(f"[XGBoost] Training completed (iteration {self.model.best_iteration})")
            return True
            
        except Exception as e:
            print(f"[XGBoost] Training failed: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> float:
        """Predict probability"""
        if self.model is None:
            return 0.5
        
        try:
            probs = self.model.predict_proba(X)[:, 1]
            epsilon = 1e-6
            probs_clip = np.clip(probs, epsilon, 1 - epsilon)
            scores = np.log(probs_clip / (1 - probs_clip))
            scores = np.clip(scores, -20, 20)
            
            calibrated = algo_platt_predict(scores, self.platt_params)
            return float(np.mean(calibrated))
        except:
            return 0.5


# ============================================================
# RANDOM FOREST COMPONENT
# ============================================================

class RandomForestComponent:
    """Random Forest model component"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.platt_params = None
        self.feature_cols = None
    
    def train(self, file_path: str) -> bool:
        """Train RF model"""
        print("\n[Random Forest] Training component...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Load data (same preprocessing as R RF code)
            df = pd.read_csv(file_path)
            df = df[df['date_id'] >= 1006].sort_values('date_id')
            
            target_col = "market_forward_excess_returns"
            exclude = ['date_id', 'forward_returns', 'risk_free_rate', target_col]
            features = [c for c in df.columns if c not in exclude]
            
            # Core factors protection
            cores = ['M4', 'P4', 'P7']
            present_cores = [f for f in cores if f in df.columns]
            
            # Missing filter
            missing = df[features].isnull().mean()
            to_drop = missing[missing > 0.20].index.tolist()
            to_drop = [c for c in to_drop if c not in present_cores]
            self.feature_cols = [f for f in features if f not in to_drop]
            
            # LOCF filling
            df_proc = df[['date_id', target_col] + self.feature_cols].copy()
            df_proc = df_proc.fillna(method='ffill').fillna(0)
            df_proc = df_proc[df_proc[target_col].notna()]
            df_proc['y'] = (df_proc[target_col] >= 0).astype(int)
            
            # Split with warmup removal
            train = df_proc[df_proc['date_id'] <= 7000]
            warmup = train['date_id'].min() + 250
            train = train[train['date_id'] > warmup]
            
            X_train = train[self.feature_cols].values
            y_train = train['y'].values
            
            # Train RF
            params = self.config['rf_params']
            max_feat = int(np.sqrt(len(self.feature_cols))) if params['mtry'] == 'sqrt' else params['mtry']
            
            self.model = RandomForestClassifier(
                n_estimators=params['ntree'],
                max_depth=params['max_depth'],
                min_samples_leaf=params['nodesize'],
                max_features=max_feat,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            self.model.fit(X_train, y_train)
            
            # Fit Platt on training
            train_probs = self.model.predict_proba(X_train)[:, 1]
            epsilon = 1e-6
            probs_clip = np.clip(train_probs, epsilon, 1 - epsilon)
            scores = np.log(probs_clip / (1 - probs_clip))
            scores = np.clip(scores, -20, 20)
            scores[np.isnan(scores) | np.isinf(scores)] = 0
            
            self.platt_params = algo_platt_fit(scores, y_train)
            
            print(f"[Random Forest] Training completed")
            return True
            
        except Exception as e:
            print(f"[Random Forest] Training failed: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> float:
        """Predict probability"""
        if self.model is None:
            return 0.5
        
        try:
            probs = self.model.predict_proba(X)[:, 1]
            epsilon = 1e-6
            probs_clip = np.clip(probs, epsilon, 1 - epsilon)
            scores = np.log(probs_clip / (1 - probs_clip))
            scores = np.clip(scores, -20, 20)
            scores[np.isnan(scores) | np.isinf(scores)] = 0
            
            calibrated = algo_platt_predict(scores, self.platt_params)
            return float(np.mean(calibrated))
        except:
            return 0.5


# ============================================================
# LOGISTIC REGRESSION COMPONENT
# ============================================================

class LogisticComponent:
    """Handmade logistic regression component"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.theta = None
        self.feature_cols = None
        self.scaler_mean = None
        self.scaler_std = None
    
    def train(self, file_path: str) -> bool:
        """Train logistic regression"""
        print("\n[Logistic] Training component...")
        
        try:
            # Load data (P and V features only)
            df = pd.read_csv(file_path)
            target_col = "market_forward_excess_returns"
            self.feature_cols = [c for c in df.columns if c.startswith('P') or c.startswith('V')]
            
            df_model = df[['date_id'] + self.feature_cols + [target_col]]
            df_model = df_model[df_model['date_id'] >= 1006]
            df_clean = df_model.dropna()
            
            X_raw = df_clean[self.feature_cols].values
            y = (df_clean[target_col] >= 0).astype(int)
            
            # Standardize
            self.scaler_mean = np.mean(X_raw, axis=0)
            self.scaler_std = np.std(X_raw, axis=0, ddof=1)
            self.scaler_std[self.scaler_std == 0] = 1.0
            X_scaled = (X_raw - self.scaler_mean) / self.scaler_std
            
            # Split
            split = int(0.8 * len(X_scaled))
            X_train = X_scaled[:split]
            y_train = y[:split].reshape(-1, 1)
            
            # Gradient descent
            m, n = X_train.shape
            X_bias = np.c_[np.ones((m, 1)), X_train]
            self.theta = np.zeros((n + 1, 1))
            
            lr = self.config['learning_rate']
            iterations = self.config['iterations']
            
            for i in range(iterations):
                h = sigmoid(X_bias @ self.theta)
                gradient = (X_bias.T @ (h - y_train)) / m
                self.theta -= lr * gradient
                
                if (i + 1) % 1000 == 0:
                    cost = -(1/m) * np.sum(y_train * np.log(h + 1e-15) + (1 - y_train) * np.log(1 - h + 1e-15))
                    print(f"  Iteration {i+1}: Cost = {cost:.6f}")
            
            print(f"[Logistic] Training completed")
            return True
            
        except Exception as e:
            print(f"[Logistic] Training failed: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> float:
        """Predict probability"""
        if self.theta is None:
            return 0.5
        
        try:
            # Standardize
            X_scaled = (X - self.scaler_mean) / self.scaler_std
            
            # Add intercept
            X_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
            
            # Predict
            probs = sigmoid(X_bias @ self.theta)
            return float(np.mean(probs))
        except:
            return 0.5


# ============================================================
# ENSEMBLE PREDICTOR
# ============================================================

class EnsembleTacticalPredictor:
    """Full ensemble with all three models"""
    
    def __init__(self):
        self.config = {
            # XGBoost config
            'xgb_params': {
                'eta': 0.01,
                'max_depth': 1,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 10,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'nrounds': 800,  # Reduced for speed
                'early_stopping': 40
            },
            
            # RF config
            'rf_params': {
                'ntree': 300,  # Reduced for speed
                'mtry': 'sqrt',
                'nodesize': 10,
                'max_depth': 5
            },
            
            # Logistic config
            'learning_rate': 0.1,
            'iterations': 3000,  # Reduced for speed
            
            # Strategy config
            'target_vol': 0.18,
            'max_leverage': 2.0,
            'prob_slope': 10
        }
        
        self.xgb = XGBoostComponent(self.config)
        self.rf = RandomForestComponent(self.config)
        self.logistic = LogisticComponent(self.config)
        
        self.prob_history = []
    
    def train(self, data_path: str):
        """Train all components"""
        print("\n" + "="*60)
        print("ENSEMBLE TACTICAL STRATEGY - Training All Components")
        print("="*60)
        
        # Train each component
        xgb_ok = self.xgb.train(data_path)
        rf_ok = self.rf.train(data_path)
        logistic_ok = self.logistic.train(data_path)
        
        # Report
        print("\n" + "="*60)
        print("Training Summary:")
        print(f"  XGBoost:   {'✓ Success' if xgb_ok else '✗ Failed'}")
        print(f"  RF:        {'✓ Success' if rf_ok else '✗ Failed'}")
        print(f"  Logistic:  {'✓ Success' if logistic_ok else '✗ Failed'}")
        print("="*60)
    
    def predict_single_step(self, test_df: pl.DataFrame) -> float:
        """Ensemble prediction"""
        test_pd = test_df.to_pandas()
        
        predictions = []
        weights = []
        
        # XGBoost prediction
        if self.xgb.model is not None:
            try:
                X_xgb = np.zeros((len(test_pd), len(self.xgb.feature_cols)))
                for i, feat in enumerate(self.xgb.feature_cols):
                    if feat in test_pd.columns:
                        X_xgb[:, i] = test_pd[feat].fillna(0)
                
                prob_xgb = self.xgb.predict(X_xgb)
                predictions.append(prob_xgb)
                weights.append(0.4)  # 40% weight
            except:
                pass
        
        # RF prediction
        if self.rf.model is not None:
            try:
                X_rf = np.zeros((len(test_pd), len(self.rf.feature_cols)))
                for i, feat in enumerate(self.rf.feature_cols):
                    if feat in test_pd.columns:
                        X_rf[:, i] = test_pd[feat].fillna(0)
                
                prob_rf = self.rf.predict(X_rf)
                predictions.append(prob_rf)
                weights.append(0.3)  # 30% weight
            except:
                pass
        
        # Logistic prediction
        if self.logistic.theta is not None:
            try:
                X_log = np.zeros((len(test_pd), len(self.logistic.feature_cols)))
                for i, feat in enumerate(self.logistic.feature_cols):
                    if feat in test_pd.columns:
                        X_log[:, i] = test_pd[feat].fillna(0)
                
                prob_log = self.logistic.predict(X_log)
                predictions.append(prob_log)
                weights.append(0.3)  # 30% weight
            except:
                pass
        
        # Ensemble
        if len(predictions) > 0:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            avg_prob = sum(p * w for p, w in zip(predictions, weights))
        else:
            avg_prob = 0.5
        
        # Update history
        self.prob_history.append(avg_prob)
        
        # Calculate adaptive center
        if len(self.prob_history) >= 60:
            center = np.median(self.prob_history[-60:])
        else:
            center = np.median(self.prob_history) if self.prob_history else 0.5
        
        # Position calculation (following R code)
        signal = 0.5 + (avg_prob - center) * self.config['prob_slope']
        base_pos = np.clip(signal, 0.2, 1.5)
        
        position = float(np.clip(base_pos, 0, self.config['max_leverage']))
        
        return position


# ============================================================
# KAGGLE API
# ============================================================

predictor = None
initialized = False


def predict(test: pl.DataFrame) -> float:
    """Main prediction function"""
    global predictor, initialized
    
    if not initialized:
        print("\n" + "="*60)
        print("INITIALIZING ENSEMBLE PREDICTOR")
        print("="*60)
        
        try:
            predictor = EnsembleTacticalPredictor()
            train_path = '/kaggle/input/hull-tactical-market-prediction/train.csv'
            
            if os.path.exists(train_path):
                predictor.train(train_path)
                initialized = True
            else:
                print("ERROR: Training file not found")
                return 0.0
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    try:
        position = predictor.predict_single_step(test)
        return position
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
