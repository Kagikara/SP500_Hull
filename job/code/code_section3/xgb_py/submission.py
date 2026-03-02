"""
Hull Tactical Market Prediction - Python Submission
Converted from R code with XGBoost, Logistic Regression, and Random Forest models
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.default_inference_server


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    """Global configuration for all models"""
    
    # Data processing
    MISSING_THRESH = 0.20
    TRAIN_END_ID = 7000
    TEST_START_ID = 7001
    TEST_END_ID = 7984
    WARMUP_PERIOD = 250
    
    # XGBoost parameters
    XGB_PARAMS = {
        'eta': 0.01,
        'max_depth': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'random_state': 42
    }
    
    # Random Forest parameters
    RF_PARAMS = {
        'n_estimators': 500,
        'max_depth': 5,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Logistic Regression parameters
    LR_PARAMS = {
        'learning_rate': 0.1,
        'iterations': 5000,
        'tolerance': 1e-6
    }
    
    # Strategy parameters
    TARGET_VOL = 0.15
    MAX_LEVERAGE = 2.0
    TREND_WINDOW = 60
    PROB_ROLL_WINDOW = 60
    OPEN_QUANTILE = 0.80
    CLOSE_QUANTILE = 0.50
    COST_RATE = 0.001


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def platt_fit(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Fit Platt Scaling parameters A and B
    P(y=1|score) = 1 / (1 + exp(A*score + B))
    """
    from scipy.optimize import minimize
    
    def objective(params):
        A, B = params
        probs = sigmoid(A * scores + B)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
        return loss
    
    result = minimize(objective, [1.0, 0.0], method='BFGS')
    return {'A': result.x[0], 'B': result.x[1]}


def platt_predict(scores: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Apply Platt Scaling transformation"""
    z = params['A'] * scores + params['B']
    return sigmoid(z)


def rolling_rank(x: np.ndarray, window: int = 250) -> np.ndarray:
    """Calculate rolling percentile rank"""
    n = len(x)
    ranks = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window + 1)
        window_data = x[start:i+1]
        if len(window_data) > 0:
            ranks[i] = np.sum(window_data <= x[i]) / len(window_data)
        else:
            ranks[i] = 0.5
    
    return ranks


# ============================================================
# DATA PROCESSOR
# ============================================================

class DataProcessor:
    """Process and prepare data for model training"""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_cols = None
        self.special_cols = None
    
    def process_training_data(self, data_path: str) -> Dict:
        """Process training data from CSV file"""
        print("[Data] Reading training data...")
        
        # Read data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        raw_data = pd.read_csv(data_path)
        
        # Define columns
        target_col = "market_forward_excess_returns"
        id_cols = ["date_id", "forward_returns", "risk_free_rate"]
        
        # Get potential features
        potential_features = [col for col in raw_data.columns 
                            if col not in [target_col] + id_cols]
        
        # Filter by date_id
        data_period = raw_data[raw_data['date_id'] >= 1006].sort_values('date_id').reset_index(drop=True)
        
        # Remove high missing rate features
        missing_ratio = data_period[potential_features].isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > self.config.MISSING_THRESH].index.tolist()
        
        # Protect core factors
        core_factors = ['M4', 'P4', 'P7']
        present_cores = [f for f in core_factors if f in data_period.columns]
        cols_to_drop = [c for c in cols_to_drop if c not in present_cores]
        
        final_feature_cols = [col for col in potential_features if col not in cols_to_drop]
        self.feature_cols = final_feature_cols
        
        print(f"  - Retained features: {len(final_feature_cols)}")
        print(f"  - Core factors: {', '.join(present_cores)}")
        
        # Select columns
        data_selected = data_period[['date_id', target_col] + final_feature_cols].copy()
        
        # Fill missing values (forward fill then fill with 0)
        data_filled = data_selected.fillna(method='ffill').fillna(0)
        
        # Remove rows with missing target
        complete_data = data_filled[data_filled[target_col].notna()].copy()
        
        # Create binary target
        complete_data['y_target'] = (complete_data[target_col] >= 0).astype(int)
        
        # Split train/test
        train_set = complete_data[complete_data['date_id'] <= self.config.TRAIN_END_ID].copy()
        test_set = complete_data[
            (complete_data['date_id'] >= self.config.TEST_START_ID) & 
            (complete_data['date_id'] <= self.config.TEST_END_ID)
        ].copy()
        
        # Remove warmup period from training
        warmup_cut = train_set['date_id'].min() + self.config.WARMUP_PERIOD
        train_set = train_set[train_set['date_id'] > warmup_cut].copy()
        
        print(f"  - Training samples: {len(train_set)}")
        print(f"  - Test samples: {len(test_set)}")
        print(f"  - Positive ratio (train): {train_set['y_target'].mean():.4f}")
        
        # Determine special columns
        val_col = 'P7' if 'P7' in final_feature_cols else 'P4'
        self.special_cols = {'momentum': 'M4', 'value': val_col}
        
        return {
            'X_train': train_set[final_feature_cols].values,
            'y_train': train_set['y_target'].values,
            'X_test': test_set[final_feature_cols].values,
            'y_test': test_set['y_target'].values,
            'raw_train_set': train_set,
            'raw_test_set': test_set,
            'feature_cols': final_feature_cols,
            'special_cols': self.special_cols
        }
    
    def process_prediction_data(self, test_df: pl.DataFrame) -> Dict:
        """Process incoming test data for prediction"""
        # Convert polars to pandas
        test_pd = test_df.to_pandas()
        
        # Ensure all features are present
        missing_feats = [f for f in self.feature_cols if f not in test_pd.columns]
        for feat in missing_feats:
            test_pd[feat] = 0
        
        # Fill missing values
        test_filled = test_pd[['date_id'] + self.feature_cols].fillna(method='ffill').fillna(0)
        
        return {
            'X_pred': test_filled[self.feature_cols].values,
            'date_ids': test_filled['date_id'].values
        }


# ============================================================
# LOGISTIC REGRESSION (HANDMADE)
# ============================================================

class LogisticRegression:
    """Handmade Logistic Regression with Gradient Descent"""
    
    def __init__(self, learning_rate: float = 0.1, iterations: int = 5000, tolerance: float = 1e-6):
        self.lr = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.theta = None
        self.cost_history = []
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column"""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """Compute log loss"""
        m = len(y)
        h = sigmoid(X @ theta)
        epsilon = 1e-15
        cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train logistic regression model"""
        print("[Logistic] Training model...")
        
        # Add intercept
        X_bias = self._add_intercept(X)
        m, n = X_bias.shape
        
        # Initialize parameters
        self.theta = np.zeros((n, 1))
        y = y.reshape(-1, 1)
        
        # Gradient descent
        for i in range(self.iterations):
            # Forward pass
            h = sigmoid(X_bias @ self.theta)
            
            # Compute gradient
            gradient = (X_bias.T @ (h - y)) / m
            
            # Update parameters
            self.theta -= self.lr * gradient
            
            # Record cost
            cost = self._compute_cost(X_bias, y, self.theta)
            self.cost_history.append(cost)
            
            # Print progress
            if (i + 1) % (self.iterations // 10) == 0:
                print(f"  Iteration {i+1}/{self.iterations}: Cost = {cost:.6f}")
            
            # Early stopping
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                print(f"  Converged at iteration {i+1}")
                break
        
        print("[Logistic] Training completed")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        X_bias = self._add_intercept(X)
        probs = sigmoid(X_bias @ self.theta)
        return probs.flatten()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


# ============================================================
# DECISION TREE (FOR RANDOM FOREST)
# ============================================================

class DecisionTree:
    """Simple decision tree for classification"""
    
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        p1 = np.mean(y)
        return 2 * p1 * (1 - p1)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray):
        """Find best split point"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        parent_gini = self._gini(y)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) <= 1:
                continue
            
            # Try median split
            threshold = np.median(feature_values)
            
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                continue
            
            # Calculate gain
            n = len(y)
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            
            gini_left = self._gini(y[left_mask])
            gini_right = self._gini(y[right_mask])
            
            weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
            gain = parent_gini - weighted_gini
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, feature_indices: np.ndarray):
        """Recursively build decision tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < 2 * self.min_samples_leaf or len(np.unique(y)) == 1:
            return {'type': 'leaf', 'value': np.mean(y)}
        
        # Find best split
        feature_idx, threshold, gain = self._find_best_split(X, y, feature_indices)
        
        if feature_idx is None or gain <= 0:
            return {'type': 'leaf', 'value': np.mean(y)}
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1, feature_indices)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1, feature_indices)
        
        return {
            'type': 'node',
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray):
        """Train decision tree"""
        self.tree = self._build_tree(X, y, 0, feature_indices)
    
    def _predict_sample(self, x: np.ndarray, node: dict) -> float:
        """Predict single sample"""
        if node['type'] == 'leaf':
            return node['value']
        
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for samples"""
        return np.array([self._predict_sample(x, self.tree) for x in X])


# ============================================================
# RANDOM FOREST (HANDMADE)
# ============================================================

class RandomForest:
    """Handmade Random Forest Classifier"""
    
    def __init__(self, n_estimators: int = 500, max_depth: int = 5, 
                 min_samples_leaf: int = 10, max_features: str = 'sqrt', random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train random forest"""
        print(f"[RF] Training {self.n_estimators} trees...")
        
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Determine max_features
        if self.max_features == 'sqrt':
            max_feat = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_feat = int(np.log2(n_features))
        else:
            max_feat = n_features
        
        print(f"  - Parameters: Trees={self.n_estimators}, Depth={self.max_depth}, mtry={max_feat}")
        
        # Train trees
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Random feature selection
            feature_indices = np.random.choice(n_features, max_feat, replace=False)
            
            # Train tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_bootstrap, y_bootstrap, feature_indices)
            self.trees.append(tree)
            
            if (i + 1) % 100 == 0:
                print(f"  - Trained {i+1}/{self.n_estimators} trees")
        
        print("[RF] Training completed")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        # Get predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Average predictions
        avg_probs = np.mean(tree_preds, axis=0)
        
        return avg_probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)


# ============================================================
# MODEL ENSEMBLE
# ============================================================

class ModelEnsemble:
    """Ensemble of XGBoost, Logistic Regression, and Random Forest"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.platt_params = {}
        self.data_processor = DataProcessor(config)
    
    def train(self, data_path: str):
        """Train all models"""
        print("\n" + "="*60)
        print("TRAINING MODEL ENSEMBLE")
        print("="*60)
        
        # Process data
        data_bundle = self.data_processor.process_training_data(data_path)
        
        X_train = data_bundle['X_train']
        y_train = data_bundle['y_train']
        X_test = data_bundle['X_test']
        y_test = data_bundle['y_test']
        
        # ========================================
        # 1. Train XGBoost
        # ========================================
        print("\n" + "-"*60)
        print("Training XGBoost Model")
        print("-"*60)
        
        try:
            import xgboost as xgb
            
            # Split for early stopping
            split_idx = int(len(X_train) * 0.8)
            X_train_split = X_train[:split_idx]
            y_train_split = y_train[:split_idx]
            X_val_split = X_train[split_idx:]
            y_val_split = y_train[split_idx:]
            
            # Train XGBoost
            xgb_model = xgb.XGBClassifier(**self.config.XGB_PARAMS)
            xgb_model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val_split, y_val_split)],
                verbose=False
            )
            
            # Get predictions on full training set
            train_probs_xgb = xgb_model.predict_proba(X_train)[:, 1]
            
            # Convert to logit scores
            epsilon = 1e-6
            train_probs_clipped = np.clip(train_probs_xgb, epsilon, 1 - epsilon)
            train_scores_xgb = np.log(train_probs_clipped / (1 - train_probs_clipped))
            
            # Fit Platt scaling on training set
            platt_xgb = platt_fit(train_scores_xgb, y_train)
            
            # Get test predictions
            test_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
            test_probs_clipped = np.clip(test_probs_xgb, epsilon, 1 - epsilon)
            test_scores_xgb = np.log(test_probs_clipped / (1 - test_probs_clipped))
            
            self.models['xgboost'] = xgb_model
            self.platt_params['xgboost'] = platt_xgb
            
            print("[XGBoost] Model trained successfully")
            print(f"  - Platt params: A={platt_xgb['A']:.4f}, B={platt_xgb['B']:.4f}")
            
        except Exception as e:
            print(f"[XGBoost] Training failed: {e}")
            self.models['xgboost'] = None
        
        # ========================================
        # 2. Train Logistic Regression
        # ========================================
        print("\n" + "-"*60)
        print("Training Logistic Regression Model")
        print("-"*60)
        
        try:
            # Standardize features for logistic regression
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train logistic regression
            lr_model = LogisticRegression(**self.config.LR_PARAMS)
            lr_model.fit(X_train_scaled, y_train)
            
            # Get predictions on training set
            train_probs_lr = lr_model.predict_proba(X_train_scaled)
            train_probs_clipped = np.clip(train_probs_lr, epsilon, 1 - epsilon)
            train_scores_lr = np.log(train_probs_clipped / (1 - train_probs_clipped))
            
            # Fit Platt scaling
            platt_lr = platt_fit(train_scores_lr, y_train)
            
            # Get test predictions
            test_probs_lr = lr_model.predict_proba(X_test_scaled)
            test_probs_clipped = np.clip(test_probs_lr, epsilon, 1 - epsilon)
            test_scores_lr = np.log(test_probs_clipped / (1 - test_probs_clipped))
            
            self.models['logistic'] = {'model': lr_model, 'scaler': scaler}
            self.platt_params['logistic'] = platt_lr
            
            print("[Logistic] Model trained successfully")
            print(f"  - Platt params: A={platt_lr['A']:.4f}, B={platt_lr['B']:.4f}")
            
        except Exception as e:
            print(f"[Logistic] Training failed: {e}")
            self.models['logistic'] = None
        
        # ========================================
        # 3. Train Random Forest
        # ========================================
        print("\n" + "-"*60)
        print("Training Random Forest Model")
        print("-"*60)
        
        try:
            # Use sklearn RandomForest for efficiency
            from sklearn.ensemble import RandomForestClassifier
            
            rf_model = RandomForestClassifier(**self.config.RF_PARAMS)
            rf_model.fit(X_train, y_train)
            
            # Get predictions on training set
            train_probs_rf = rf_model.predict_proba(X_train)[:, 1]
            train_probs_clipped = np.clip(train_probs_rf, epsilon, 1 - epsilon)
            train_scores_rf = np.log(train_probs_clipped / (1 - train_probs_clipped))
            
            # Fit Platt scaling
            platt_rf = platt_fit(train_scores_rf, y_train)
            
            # Get test predictions
            test_probs_rf = rf_model.predict_proba(X_test)[:, 1]
            test_probs_clipped = np.clip(test_probs_rf, epsilon, 1 - epsilon)
            test_scores_rf = np.log(test_probs_clipped / (1 - test_probs_clipped))
            
            self.models['random_forest'] = rf_model
            self.platt_params['random_forest'] = platt_rf
            
            print("[Random Forest] Model trained successfully")
            print(f"  - Platt params: A={platt_rf['A']:.4f}, B={platt_rf['B']:.4f}")
            
        except Exception as e:
            print(f"[Random Forest] Training failed: {e}")
            self.models['random_forest'] = None
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        return data_bundle
    
    def predict(self, test_df: pl.DataFrame) -> float:
        """Generate ensemble prediction for a single time step"""
        
        # Process test data
        pred_bundle = self.data_processor.process_prediction_data(test_df)
        X_pred = pred_bundle['X_pred']
        
        # Collect predictions from all models
        predictions = []
        weights = []
        
        # XGBoost prediction
        if self.models.get('xgboost') is not None:
            try:
                probs = self.models['xgboost'].predict_proba(X_pred)[:, 1]
                epsilon = 1e-6
                probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
                scores = np.log(probs_clipped / (1 - probs_clipped))
                scores = np.clip(scores, -20, 20)
                
                calibrated_probs = platt_predict(scores, self.platt_params['xgboost'])
                avg_prob = float(np.mean(calibrated_probs))
                
                predictions.append(avg_prob)
                weights.append(0.4)  # XGBoost gets 40% weight
            except Exception as e:
                print(f"[Warning] XGBoost prediction failed: {e}")
        
        # Logistic Regression prediction
        if self.models.get('logistic') is not None:
            try:
                scaler = self.models['logistic']['scaler']
                model = self.models['logistic']['model']
                
                X_scaled = scaler.transform(X_pred)
                probs = model.predict_proba(X_scaled)
                epsilon = 1e-6
                probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
                scores = np.log(probs_clipped / (1 - probs_clipped))
                scores = np.clip(scores, -20, 20)
                
                calibrated_probs = platt_predict(scores, self.platt_params['logistic'])
                avg_prob = float(np.mean(calibrated_probs))
                
                predictions.append(avg_prob)
                weights.append(0.3)  # Logistic gets 30% weight
            except Exception as e:
                print(f"[Warning] Logistic prediction failed: {e}")
        
        # Random Forest prediction
        if self.models.get('random_forest') is not None:
            try:
                probs = self.models['random_forest'].predict_proba(X_pred)[:, 1]
                epsilon = 1e-6
                probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
                scores = np.log(probs_clipped / (1 - probs_clipped))
                scores = np.clip(scores, -20, 20)
                
                calibrated_probs = platt_predict(scores, self.platt_params['random_forest'])
                avg_prob = float(np.mean(calibrated_probs))
                
                predictions.append(avg_prob)
                weights.append(0.3)  # RF gets 30% weight
            except Exception as e:
                print(f"[Warning] Random Forest prediction failed: {e}")
        
        # Ensemble prediction (weighted average)
        if len(predictions) > 0:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Weighted average
            final_prediction = sum(p * w for p, w in zip(predictions, weights))
            
            # Convert to position: probability > 0.5 → long (1.0), else stay out (0.0)
            position = 1.0 if final_prediction > 0.5 else 0.0
            
            return position
        else:
            # No models available, return neutral position
            return 0.0


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

# Global model ensemble
model_ensemble = None
is_initialized = False


def predict(test: pl.DataFrame) -> float:
    """
    Main prediction function called by Kaggle API
    
    Args:
        test: Polars DataFrame with test features
        
    Returns:
        float: Position (0.0 for no position, 1.0 for long position)
    """
    global model_ensemble, is_initialized
    
    # Initialize and train models on first call
    if not is_initialized:
        print("\n" + "="*60)
        print("INITIALIZING MODEL ENSEMBLE")
        print("="*60)
        
        # Path to training data
        train_data_path = '/kaggle/input/hull-tactical-market-prediction/train.csv'
        
        # Check if file exists
        if not os.path.exists(train_data_path):
            print(f"Warning: Training file not found at {train_data_path}")
            print("Returning default position of 0.0")
            return 0.0
        
        # Initialize ensemble
        config = Config()
        model_ensemble = ModelEnsemble(config)
        
        # Train models
        try:
            model_ensemble.train(train_data_path)
            is_initialized = True
            print("\nModel initialization completed successfully!")
        except Exception as e:
            print(f"Error during model initialization: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    # Make prediction
    try:
        position = model_ensemble.predict(test)
        return position
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


# ============================================================
# KAGGLE API SETUP
# ============================================================

# Create inference server
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

# Run inference
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
