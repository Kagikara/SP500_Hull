"""
Hull Tactical Market Prediction - Random Forest Version
Faithful translation of R code with hand-made RF logic
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.default_inference_server


# ============================================================
# ALGORITHM LIBRARY
# ============================================================

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


def algo_rolling_rank(x: np.ndarray, window: int = 250) -> np.ndarray:
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
# DATA PROCESSOR (from 01_data_processor_rf.R)
# ============================================================

class DataProcessorRF:
    """
    RF data processor with LOCF filling and core factor protection
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_cols = None
        self.special_cols = None
    
    def process_data(self, file_path: str) -> Dict:
        """Process data following RF R code logic"""
        print("[Data-RF] Reading data:", file_path)
        
        raw_data = pd.read_csv(file_path)
        
        # Core factors (protected from removal)
        core_factors = ['M4', 'P4', 'P7']
        present_cores = [f for f in core_factors if f in raw_data.columns]
        
        # Feature selection
        target_col = "market_forward_excess_returns"
        id_cols = ["date_id", "forward_returns", "risk_free_rate"]
        potential_features = [col for col in raw_data.columns 
                            if col not in [target_col] + id_cols]
        
        # Missing rate filter (but protect core factors)
        missing_ratio = raw_data[potential_features].isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > self.config['missing_thresh']].index.tolist()
        cols_to_drop = [c for c in cols_to_drop if c not in present_cores]
        final_feature_cols = [col for col in potential_features if col not in cols_to_drop]
        
        print(f"[Data-RF] Retained {len(final_feature_cols)} features")
        print(f"[Data-RF] Core factors: {', '.join(present_cores)}")
        
        # Select and sort data
        data_selected = raw_data[['date_id', target_col] + final_feature_cols].sort_values('date_id').copy()
        
        # LOCF filling (forward fill)
        data_filled = data_selected.fillna(method='ffill')
        
        # Remaining NA fill with 0 (RF can handle)
        data_filled[final_feature_cols] = data_filled[final_feature_cols].fillna(0)
        
        # Remove missing target
        complete_data = data_filled[data_filled[target_col].notna()].copy()
        
        # Time series alignment check
        if not complete_data['date_id'].is_monotonic_increasing:
            raise ValueError("date_id is not strictly increasing!")
        
        # Create binary target
        complete_data['y_target'] = (complete_data[target_col] >= 0).astype(int)
        
        # Train/test split
        train_set = complete_data[complete_data['date_id'] <= self.config['train_end_id']].copy()
        test_set = complete_data[
            (complete_data['date_id'] >= self.config['test_start_id']) &
            (complete_data['date_id'] <= self.config['test_end_id'])
        ].copy()
        
        # Remove warmup period (first 250 days)
        warmup_cut = train_set['date_id'].min() + 250
        train_set = train_set[train_set['date_id'] > warmup_cut].copy()
        
        print(f"[Data-RF] Train samples: {len(train_set)}, Test samples: {len(test_set)}")
        print(f"[Data-RF] Positive ratio: {train_set['y_target'].mean():.4f}")
        
        # Determine value column
        val_col = 'P7' if 'P7' in final_feature_cols else 'P4'
        self.special_cols = {'momentum': 'M4', 'value': val_col}
        self.feature_cols = final_feature_cols
        
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


# ============================================================
# MODEL TRAINER (from 02_model_trainer_rf_handmade.R)
# ============================================================

class RFModelTrainer:
    """
    Random Forest trainer using sklearn for efficiency
    Key: Fit Platt Scaling on TRAINING set only
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.platt_params = None
    
    def train(self, data_bundle: Dict) -> Dict:
        """Train RF model with proper calibration"""
        print("[Model-RF] Starting RF training...")
        
        from sklearn.ensemble import RandomForestClassifier
        
        X_train = data_bundle['X_train']
        y_train = data_bundle['y_train']
        X_test = data_bundle['X_test']
        
        params = self.config['rf_params']
        
        # Determine mtry
        n_features = X_train.shape[1]
        if params['mtry'] == 'sqrt':
            max_features = int(np.sqrt(n_features))
        else:
            max_features = params['mtry']
        
        print(f"[Model-RF] Parameters: Trees={params['ntree']}, Depth={params['max_depth']}, mtry={max_features}")
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=params['ntree'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['nodesize'],
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        print("[Model-RF] Training completed")
        
        # Feature importance
        feature_names = data_bundle['feature_cols']
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n[Model-RF] Top 10 Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # ============================================================
        # KEY: Fit Platt Scaling on TRAINING set (anti-leakage)
        # ============================================================
        print("\n[Model-RF] Fitting Platt Scaling on training set...")
        
        train_probs = self.model.predict_proba(X_train)[:, 1]
        
        # Convert to logit scores
        epsilon = 1e-6
        train_probs_clipped = np.clip(train_probs, epsilon, 1 - epsilon)
        train_scores_logit = np.log(train_probs_clipped / (1 - train_probs_clipped))
        train_scores_logit = np.clip(train_scores_logit, -20, 20)
        train_scores_logit[np.isnan(train_scores_logit) | np.isinf(train_scores_logit)] = 0
        
        # Fit Platt parameters on training data
        self.platt_params = algo_platt_fit(train_scores_logit, y_train)
        print(f"  - Platt A: {self.platt_params['A']:.4f}, B: {self.platt_params['B']:.4f}")
        
        # ============================================================
        # Generate test predictions (for backtest only)
        # ============================================================
        print("\n[Model-RF] Generating test predictions...")
        test_probs = self.model.predict_proba(X_test)[:, 1]
        
        # Clean predictions
        test_probs = np.clip(test_probs, 0.01, 0.99)
        test_probs[np.isnan(test_probs)] = 0.5
        
        # Convert to logit scores
        test_probs_clipped = np.clip(test_probs, epsilon, 1 - epsilon)
        test_scores_logit = np.log(test_probs_clipped / (1 - test_probs_clipped))
        test_scores_logit = np.clip(test_scores_logit, -20, 20)
        test_scores_logit[np.isnan(test_scores_logit) | np.isinf(test_scores_logit)] = 0
        
        print(f"  - Test logit range: [{test_scores_logit.min():.4f}, {test_scores_logit.max():.4f}]")
        
        return {
            'model': self.model,
            'test_scores': test_scores_logit,
            'platt_params': self.platt_params,
            'feature_importance': importance_df,
            'model_type': 'random_forest'
        }


# ============================================================
# STRATEGY ENGINE (from 03_strategy_engine_rf.R)
# ============================================================

class StrategyEngineRF:
    """
    RF Strategy with strict anti-leakage:
    - Use training Platt params
    - Lag all dynamic thresholds
    - Lag volatility calculations
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def run_backtest(self, data_bundle: Dict, model_bundle: Dict) -> Dict:
        """Execute strategy backtest with full anti-leakage measures"""
        print("[Strategy-RF] Starting anti-leakage backtest...")
        
        # Parameters
        p_target_vol = self.config.get('target_vol', 0.15)
        p_max_leverage = self.config.get('max_leverage', 2.0)
        p_trend_window = self.config.get('trend_window', 60)
        p_roll_window = self.config.get('prob_roll_window', 60)
        p_open_q = self.config.get('open_quantile', 0.80)
        p_close_q = self.config.get('close_quantile', 0.50)
        p_cost_rate = self.config.get('cost_rate', 0.001)
        
        # Data
        test_set = data_bundle['raw_test_set'].copy()
        test_scores = model_bundle['test_scores']
        test_scores = np.clip(test_scores, -20, 20)
        test_scores[np.isnan(test_scores) | np.isinf(test_scores)] = 0
        
        # ============================================================
        # Use TRAINING Platt params (key anti-leakage fix)
        # ============================================================
        platt_params = model_bundle['platt_params']
        print(f"[Strategy-RF] Using training Platt params: A={platt_params['A']:.4f}, B={platt_params['B']:.4f}")
        
        probs = algo_platt_predict(test_scores, platt_params)
        probs[np.isnan(probs) | np.isinf(probs)] = 0.5
        probs = np.clip(probs, 0.01, 0.99)
        
        test_set['pred_prob'] = probs
        n = len(probs)
        
        print(f"  - Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
        
        # ============================================================
        # Dynamic thresholds with strict lag
        # ============================================================
        print("[Strategy-RF] Calculating dynamic thresholds (with lag)...")
        
        # Rolling quantiles (align right, then lag)
        roll_q_open = np.zeros(n)
        roll_q_close = np.zeros(n)
        
        for i in range(n):
            start = max(0, i - p_roll_window + 1)
            window_probs = probs[start:i+1]
            roll_q_open[i] = np.quantile(window_probs, p_open_q)
            roll_q_close[i] = np.quantile(window_probs, p_close_q)
        
        # Lag by 1 period
        thresh_open = np.concatenate([[roll_q_open[0]], roll_q_open[:-1]])
        thresh_close = np.concatenate([[roll_q_close[0]], roll_q_close[:-1]])
        
        # Fill warmup period
        warmup_end = p_roll_window + 1
        thresh_open[:warmup_end] = np.quantile(probs[:warmup_end], p_open_q)
        thresh_close[:warmup_end] = np.quantile(probs[:warmup_end], p_close_q)
        
        # ============================================================
        # Hysteresis signals (state machine)
        # ============================================================
        print("[Strategy-RF] Generating hysteresis signals...")
        
        signals = np.zeros(n, dtype=int)
        state = 0
        
        for i in range(n):
            if state == 0:
                if probs[i] > thresh_open[i]:
                    state = 1
            else:
                if probs[i] < thresh_close[i]:
                    state = 0
            signals[i] = state
        
        test_set['signal_rf'] = signals
        
        # ============================================================
        # Risk factors (with lag)
        # ============================================================
        print("[Strategy-RF] Calculating risk factors (with lag)...")
        
        returns = test_set['market_forward_excess_returns'].values
        
        # Lagged returns for volatility
        past_returns = np.concatenate([[0], returns[:-1]])
        
        # Rolling volatility
        current_vol = np.zeros(n)
        for i in range(n):
            start = max(0, i - 19)
            window_returns = past_returns[start:i+1]
            if len(window_returns) > 1:
                current_vol[i] = np.std(window_returns, ddof=1) * np.sqrt(252)
            else:
                current_vol[i] = p_target_vol
        
        # Lag volatility by 1
        current_vol = np.concatenate([[p_target_vol], current_vol[:-1]])
        current_vol[current_vol <= 0] = p_target_vol
        
        # Trend (using lagged price)
        pseudo_price = np.cumprod(1 + past_returns)
        
        ma_trend = np.zeros(n)
        for i in range(n):
            start = max(0, i - p_trend_window + 1)
            ma_trend[i] = np.mean(pseudo_price[start:i+1])
        
        # Lag trend
        ma_trend = np.concatenate([[pseudo_price[0]], ma_trend[:-1]])
        is_bull = (pseudo_price > ma_trend).astype(int)
        
        # ============================================================
        # Interaction filter (M4 x P7)
        # ============================================================
        col_m = data_bundle['special_cols']['momentum']
        col_p = data_bundle['special_cols']['value']
        
        rank_m = algo_rolling_rank(test_set[col_m].values, window=250)
        rank_p = algo_rolling_rank(test_set[col_p].values, window=250)
        
        filter_mp = np.ones(n)
        filter_mp[rank_m < 0.2] = 0.0
        
        mask_boost = (rank_m > 0.7) & (rank_p < 0.4)
        filter_mp[mask_boost] = 2.0
        
        mask_bubble = rank_p > 0.9
        filter_mp[mask_bubble & ~mask_boost] = 0.5
        
        # ============================================================
        # Final position synthesis
        # ============================================================
        vol_scaler = p_target_vol / current_vol
        trend_mult = np.where(is_bull == 1, 1.0, 0.6)
        
        raw_pos = signals * filter_mp * vol_scaler * trend_mult
        position = np.clip(raw_pos, 0, p_max_leverage)
        position[np.isnan(position)] = 0
        
        test_set['position'] = position
        
        # Trading costs
        turnover = np.concatenate([[0], np.abs(np.diff(position))])
        gross_return = position * returns
        net_return = gross_return - (turnover * p_cost_rate)
        
        test_set['turnover'] = turnover
        test_set['net_return'] = net_return
        
        # ============================================================
        # Performance metrics
        # ============================================================
        def calc_metrics(ret):
            ret = ret[~np.isnan(ret)]
            total = np.prod(1 + ret) - 1
            ann = (1 + total) ** (252 / len(ret)) - 1
            vol = np.std(ret, ddof=1) * np.sqrt(252)
            sharpe = ann / vol if vol > 0 else 0
            cum = np.cumprod(1 + ret)
            dd = np.min(cum / np.maximum.accumulate(cum) - 1)
            return {'Ann': ann, 'Sharpe': sharpe, 'DD': dd, 'Vol': vol}
        
        res = calc_metrics(net_return)
        
        print("\n" + "="*50)
        print("RF Strategy Performance (Anti-Leakage + Costs)")
        print("="*50)
        print(f"  Trading Cost    : {p_cost_rate:.4f} (one-way)")
        print(f"  Annual Return   : {res['Ann']*100:.2f}%")
        print(f"  Annual Vol      : {res['Vol']*100:.2f}%")
        print(f"  Sharpe Ratio    : {res['Sharpe']:.4f}")
        print(f"  Max Drawdown    : {res['DD']*100:.2f}%")
        print(f"  Avg Turnover    : {np.mean(turnover):.4f}")
        print("="*50)
        
        return {
            'results': test_set,
            'metrics': res,
            'platt_params': platt_params
        }


# ============================================================
# MAIN PREDICTOR
# ============================================================

class RandomForestTacticalPredictor:
    """Complete RF tactical predictor with full R code strategy"""
    
    def __init__(self):
        self.config = {
            'missing_thresh': 0.20,
            'train_end_id': 7000,
            'test_start_id': 7001,
            'test_end_id': 7984,
            
            'rf_params': {
                'ntree': 500,
                'mtry': 'sqrt',
                'nodesize': 10,
                'max_depth': 5
            },
            
            'target_vol': 0.15,
            'max_leverage': 2.0,
            'trend_window': 60,
            'prob_roll_window': 60,
            'open_quantile': 0.80,
            'close_quantile': 0.50,
            'cost_rate': 0.001
        }
        
        self.model = None
        self.platt_params = None
        self.feature_cols = None
        self.special_cols = None
        
        # Online prediction state
        self.prob_history = []
        self.state = 0  # Hysteresis state
    
    def train(self, data_path: str):
        """Train complete system"""
        print("\n" + "="*60)
        print("Random Forest Tactical Strategy - Training")
        print("="*60)
        
        # Stage 1: Data processing
        print("\n>>> Stage I: Data Processing")
        processor = DataProcessorRF(self.config)
        data_bundle = processor.process_data(data_path)
        self.feature_cols = data_bundle['feature_cols']
        self.special_cols = data_bundle['special_cols']
        
        # Stage 2: Model training
        print("\n>>> Stage II: RF Training")
        trainer = RFModelTrainer(self.config)
        model_bundle = trainer.train(data_bundle)
        self.model = model_bundle['model']
        self.platt_params = model_bundle['platt_params']
        
        # Stage 3: Strategy backtest
        print("\n>>> Stage III: Strategy Backtest")
        engine = StrategyEngineRF(self.config)
        backtest_results = engine.run_backtest(data_bundle, model_bundle)
        
        print("\n" + "="*60)
        print("Training Completed")
        print("="*60)
    
    def predict_single_step(self, test_df: pl.DataFrame) -> float:
        """Predict position for single time step"""
        test_pd = test_df.to_pandas()
        
        # Ensure features
        for feat in self.feature_cols:
            if feat not in test_pd.columns:
                test_pd[feat] = 0
        
        X_pred = test_pd[self.feature_cols].values
        
        # Get RF probability
        probs = self.model.predict_proba(X_pred)[:, 1]
        
        # Convert to logit
        epsilon = 1e-6
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        scores = np.log(probs_clipped / (1 - probs_clipped))
        scores = np.clip(scores, -20, 20)
        
        # Apply Platt Scaling
        calibrated_probs = algo_platt_predict(scores, self.platt_params)
        avg_prob = float(np.mean(calibrated_probs))
        
        # Update history
        self.prob_history.append(avg_prob)
        
        # Dynamic thresholds
        if len(self.prob_history) >= self.config['prob_roll_window']:
            recent = self.prob_history[-self.config['prob_roll_window']:]
            thresh_open = np.quantile(recent, self.config['open_quantile'])
            thresh_close = np.quantile(recent, self.config['close_quantile'])
        else:
            thresh_open = 0.99
            thresh_close = 0.50
        
        # Hysteresis signal
        if self.state == 0:
            if avg_prob > thresh_open:
                self.state = 1
        else:
            if avg_prob < thresh_close:
                self.state = 0
        
        signal = float(self.state)
        
        # Conservative position (no historical data for vol/trend)
        position = signal * 1.0
        position = float(np.clip(position, 0, self.config['max_leverage']))
        
        return position


# ============================================================
# KAGGLE API
# ============================================================

predictor = None
initialized = False


def predict(test: pl.DataFrame) -> float:
    global predictor, initialized
    
    if not initialized:
        print("\n" + "="*60)
        print("INITIALIZING RF TACTICAL PREDICTOR")
        print("="*60)
        
        try:
            predictor = RandomForestTacticalPredictor()
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
