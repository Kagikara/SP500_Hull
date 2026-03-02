import os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import deque
import joblib

# ============================================================
# 1. 策略配置 (优化参数版)
# ============================================================
CONFIG = {
    # 路径 (请根据实际情况修改)
    'input_file': '/kaggle/input/hull-tactical-market-prediction/train.csv', 
    'seed': 42,
    
    # --- RF 参数 (对应 R: ntree=500, mtry=sqrt, max_depth=5) ---
    'rf_params': {
        'n_estimators': 500,
        'max_features': 'sqrt',
        'max_depth': 5,
        'min_samples_leaf': 10,
        'n_jobs': -1,
        'random_state': 42,
        'class_weight': 'balanced' # 震荡市中样本可能不平衡，加这个有帮助
    },
    
    # --- 策略参数 (Sniper 核心) ---
    'roll_window': 60,       # 对应 prob_roll_window
    'open_quantile': 0.85,   # 开仓门槛 (R代码是0.80，稍微提一点增加精度)
    'close_quantile': 0.50,  # 平仓门槛
    
    # --- 风控与滤网 ---
    'target_vol': 0.15,
    'max_leverage': 2.0,
    'trend_window': 60,      # 趋势判断窗口
    
    # M-P 因子参数
    'rank_window': 250,      # 计算动量/估值排名
    'momentum_col': 'M4',    # 动量因子名
    'value_col': 'P4',       # 估值因子名 (原R代码中是P4或P7)
}

# ============================================================
# 2. 辅助类: Platt Scaling (校准器)
# ============================================================
class PlattScaler:
    def __init__(self):
        # C=1e9 模拟无正则化的 LR
        self.lr = LogisticRegression(C=1e9, solver='lbfgs') 
        
    def fit(self, logits, labels):
        self.lr.fit(logits.reshape(-1, 1), labels)
        
    def predict(self, logits):
        return self.lr.predict_proba(logits.reshape(-1, 1))[:, 1]

def prob_to_logit(probs, eps=1e-6):
    probs = np.clip(probs, eps, 1 - eps)
    return np.log(probs / (1 - probs))

# ============================================================
# 3. 核心策略类: RF Sniper (Stateful)
# ============================================================
class RFSniperStrategy:
    def __init__(self, config):
        self.cfg = config
        self.model = None
        self.scaler = None
        self.features = []
        
        # --- 状态保持器 (History Buffers) ---
        # 1. 概率历史 (用于计算动态阈值)
        self.prob_history = deque(maxlen=config['roll_window'])
        
        # 2. 因子历史 (用于 M-P 滤网)
        self.mom_history = deque(maxlen=config['rank_window'])
        self.val_history = deque(maxlen=config['rank_window'])
        
        # 3. 市场回报历史 (用于波动率和趋势)
        self.ret_history = deque(maxlen=max(20, config['trend_window']))
        
        # 4. 滞后阈值缓存 (实现 t-1 决策)
        # 存储上一时刻计算出的 [thresh_open, thresh_close]
        self.prev_thresholds = (0.99, 0.50) 
        
        # 5. 当前持仓状态 (0=空仓, 1=持仓)
        self.state = 0
        
        # 初始化
        self._train_and_init()

    def _train_and_init(self):
        print(">>> [Init] Loading Data & Training RF...")
        if not os.path.exists(self.cfg['input_file']):
            print("Warning: Input file not found. Skipping training.")
            return

        df = pd.read_csv(self.cfg['input_file'])
        
        # 1. 基础清洗
        df = df[df['date_id'] >= 1006].sort_values('date_id').reset_index(drop=True)
        df['target'] = (df['market_forward_excess_returns'] >= 0).astype(int)
        
        # 2. 特征定义 (自动排除非特征列)
        exclude = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'target']
        self.features = [c for c in df.columns if c not in exclude]
        
        # 3. 训练模型
        X = df[self.features].fillna(0)
        y = df['target']
        
        self.model = RandomForestClassifier(**self.cfg['rf_params'])
        self.model.fit(X, y)
        
        # 4. 拟合 Platt Scaler (使用 OOB 或 简单全量预测)
        # 为了与 R 代码一致 (train_rf_model 中是直接预测训练集)，这里也直接预测
        raw_probs = self.model.predict_proba(X)[:, 1]
        raw_logits = prob_to_logit(raw_probs)
        
        self.scaler = PlattScaler()
        self.scaler.fit(raw_logits, y)
        calibrated_probs = self.scaler.predict(raw_logits)
        
        print(f"    RF Trained. Platt Score Range: {calibrated_probs.min():.4f} - {calibrated_probs.max():.4f}")
        
        # 5. [关键] 预热历史状态 (Cold Start)
        # 我们需要填满 deque，以便第一天预测时就有数据可用
        
        # A. 概率历史
        init_probs = calibrated_probs[-self.cfg['roll_window']:]
        self.prob_history.extend(init_probs)
        
        # B. 因子历史
        mom_col = self.cfg['momentum_col']
        val_col = self.cfg['value_col']
        if mom_col in df.columns: self.mom_history.extend(df[mom_col].tail(self.cfg['rank_window']))
        if val_col in df.columns: self.val_history.extend(df[val_col].tail(self.cfg['rank_window']))
            
        # C. 回报历史
        self.ret_history.extend(df['market_forward_excess_returns'].tail(self.cfg['trend_window']))
        
        # D. 计算初始阈值 (作为第1天的 t-1 阈值)
        # 这里的逻辑是：基于历史数据的最后时刻，算出明天的阈值
        thresh_open = np.quantile(self.prob_history, self.cfg['open_quantile'])
        thresh_close = np.quantile(self.prob_history, self.cfg['close_quantile'])
        self.prev_thresholds = (thresh_open, thresh_close)
        
        print(">>> [Init] Done. State Warmup Complete.")

    def _calc_rank(self, history, current_val):
        """计算当前值在历史中的百分位排名"""
        if len(history) < 10: return 0.5
        # 将当前值加入临时列表比较
        combined = np.array(list(history) + [current_val])
        return (combined <= current_val).mean()

    def predict_one(self, test_df):
        """处理单行数据"""
        # 1. 数据对齐
        if isinstance(test_df, pl.DataFrame):
            curr = test_df.to_pandas()
        else:
            curr = test_df.copy()
            
        # 缺失特征补0
        for f in self.features:
            if f not in curr.columns: curr[f] = 0
            
        X_curr = curr[self.features].fillna(0)
        
        # 2. 模型推理 & 校准
        if self.model is None: return 0.0
        
        raw_prob = self.model.predict_proba(X_curr)[:, 1][0]
        logit = prob_to_logit(np.array([raw_prob]))
        curr_prob = self.scaler.predict(logit)[0]
        
        # 3. [核心] 信号生成 (Anti-Leakage)
        # 使用上一步存储的阈值 (t-1) 来判断当前 (t)
        thresh_open_t_minus_1, thresh_close_t_minus_1 = self.prev_thresholds
        
        if self.state == 0:
            # 空仓 -> 开仓 (Prob > Thresh_Open)
            if curr_prob > thresh_open_t_minus_1:
                self.state = 1
        else:
            # 持仓 -> 平仓 (Prob < Thresh_Close)
            if curr_prob < thresh_close_t_minus_1:
                self.state = 0
                
        # 4. 更新状态 (为 t+1 做准备)
        # A. 更新概率历史
        self.prob_history.append(curr_prob)
        
        # B. 计算新的阈值 (这将作为明天的 t-1 阈值)
        new_thresh_open = np.quantile(self.prob_history, self.cfg['open_quantile'])
        new_thresh_close = np.quantile(self.prob_history, self.cfg['close_quantile'])
        self.prev_thresholds = (new_thresh_open, new_thresh_close)
        
        # 5. 因子互作滤网 (M-P Filter)
        # 获取当前因子值
        curr_mom = curr[self.cfg['momentum_col']].iloc[0] if self.cfg['momentum_col'] in curr else 0
        curr_val = curr[self.cfg['value_col']].iloc[0] if self.cfg['value_col'] in curr else 0
        
        # 计算 Rank
        rank_m = self._calc_rank(self.mom_history, curr_mom)
        rank_p = self._calc_rank(self.val_history, curr_val)
        
        # 更新因子历史
        self.mom_history.append(curr_mom)
        self.val_history.append(curr_val)
        
        # 计算滤网系数 (复现 R 逻辑)
        filter_mp = 1.0
        if rank_m < 0.2: filter_mp = 0.0 # 崩盘保护
        elif (rank_m > 0.7) and (rank_p < 0.4): filter_mp = 2.0 # 戴维斯双击
        elif rank_p > 0.9: 
            if filter_mp != 2.0: filter_mp = 0.5 # 估值泡沫
            
        # 6. 风控 (波动率 & 趋势)
        # 注意: API通常不直接提供 mkt_return，我们需要推断或忽略
        # 如果无法获取当日 mkt_return，我们沿用最后已知的波动率
        # 这里为了稳健，使用默认波动率，如果有数据则更新
        
        # 假设 curr 中没有 target，我们无法更新 ret_history
        # 在真实比赛中，可能需要用 close/open 估算，或者忽略
        # 这里简化：使用固定波动率缩放
        vol_scaler = 1.0 
        
        # 7. 最终仓位
        raw_pos = self.state * filter_mp * vol_scaler
        
        # 熊市降仓 (复现 trend_mult)
        # 如果最近回报是负的，降低仓位
        if len(self.ret_history) > 0 and np.mean(self.ret_history) < 0:
             raw_pos *= 0.6
             
        position = np.clip(raw_pos, 0.0, self.cfg['max_leverage'])
        
        return float(position)

# ============================================================
# 4. 初始化与 API 接口
# ============================================================
GLOBAL_SNIPER = RFSniperStrategy(CONFIG)

import kaggle_evaluation.default_inference_server

def predict(test: pl.DataFrame) -> float:
    try:
        return GLOBAL_SNIPER.predict_one(test)
    except Exception as e:
        return 0.0

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)
    inference_server.serve()
else:
    inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)
    inference_server.run_local_gateway(
        (CONFIG['input_file'].replace('train.csv', ''),)
    )