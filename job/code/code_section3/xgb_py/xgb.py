import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ============================================================
# 1. 全局配置 (CONFIG)
# ============================================================
CONFIG = {
    'input_file': 'C:/Users/kagirasu/works_H/ml_HULL/job/code/code_section3/proj_svm/data/train.csv',  # 请修改为你的实际路径
    'seed': 42,
    
    # 数据处理
    'missing_thresh': 0.20,
    'train_end_id': 7000,
    'test_start_id': 7001,
    'test_end_id': 7984,
    
    # XGB 参数
    'xgb_params': {
        'n_estimators': 1500,        # 对应 R 的 nrounds
        'learning_rate': 0.01,       # 对应 R 的 eta
        'max_depth': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'objective': 'binary:logistic',
        'n_jobs': -1,
        'random_state': 42,
        'tree_method': 'hist'        # 加速训练
    },
    'early_stopping_rounds': 50,
    
    # 策略参数
    'target_vol': 0.20,
    'max_leverage': 2.0,
    'prob_roll_window': 60,   # 自适应窗口
    'prob_slope': 10.0,       # 信号放大倍数
    'trend_window': 60,       # 牛熊判断窗口
    'transaction_cost': 0.001
}

# ============================================================
# 2. 辅助函数: Platt Scaling
# ============================================================
class PlattScaler:
    def __init__(self):
        self.lr = LogisticRegression(C=1e9, solver='lbfgs') # C很大表示弱正则化，接近纯LR
        
    def fit(self, logits, labels):
        # Sklearn 需要 (n_samples, 1) 的形状
        self.lr.fit(logits.reshape(-1, 1), labels)
        
    def predict(self, logits):
        # 返回概率 (class 1)
        return self.lr.predict_proba(logits.reshape(-1, 1))[:, 1]
    
    @property
    def coefs(self):
        return self.lr.coef_[0][0], self.lr.intercept_[0]

def prob_to_logit(probs, eps=1e-6):
    probs = np.clip(probs, eps, 1 - eps)
    return np.log(probs / (1 - probs))

# ============================================================
# 3. 数据处理器 (Data Processor)
# ============================================================
def process_data(file_path, config):
    print(f"[Data] Reading {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return None

    # 1. 基础筛选
    target_col = "market_forward_excess_returns"
    id_cols = ["date_id", "forward_returns", "risk_free_rate"]
    
    # 仅保留有效时间段
    df = df[df['date_id'] >= 1006].sort_values('date_id').reset_index(drop=True)
    
    # 2. 特征筛选 (基于训练集统计缺失率)
    train_mask = df['date_id'] <= config['train_end_id']
    train_subset = df[train_mask]
    
    potential_feats = [c for c in df.columns if c not in [target_col] + id_cols]
    
    # 计算缺失率
    missing_ratio = train_subset[potential_feats].isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > config['missing_thresh']].index.tolist()
    final_feats = [f for f in potential_feats if f not in cols_to_drop]
    
    print(f"  - Features Dropped: {len(cols_to_drop)}")
    print(f"  - Features Kept: {len(final_feats)}")
    
    # 3. 清洗与构建 Target
    # 移除 Target 为 NA 的行
    df = df.dropna(subset=[target_col]).copy()
    
    # 构造 0/1 标签
    df['y_target'] = (df[target_col] >= 0).astype(int)
    
    # 4. 切分
    train_df = df[df['date_id'] <= config['train_end_id']].copy()
    test_df = df[(df['date_id'] >= config['test_start_id']) & 
                 (df['date_id'] <= config['test_end_id'])].copy()
    
    return {
        'X_train': train_df[final_feats],
        'y_train': train_df['y_target'],
        'X_test': test_df[final_feats],
        'y_test': test_df['y_target'],
        'raw_train': train_df,
        'raw_test': test_df,
        'features': final_feats
    }

# ============================================================
# 4. 模型训练器 (Model Trainer - CV & Calib)
# ============================================================
def train_xgb_model(data_bundle, config):
    print("[Model-XGB] Starting Training (5-Fold CV)...")
    
    X_train = data_bundle['X_train']
    y_train = data_bundle['y_train']
    X_test = data_bundle['X_test']
    
    # 1. 交叉验证获取 OOF 预测
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['seed'])
    oof_probs = np.zeros(len(y_train))
    best_iters = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**config['xgb_params'], 
                                  early_stopping_rounds=config['early_stopping_rounds'])
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        # 记录最佳轮数
        # 注意: XGBClassifier 的 best_iteration 是从 0 开始的索引，通常不需要加1，除非用于 slice
        best_iters.append(model.best_iteration)
        
        # 预测验证集
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        print(f"  - Fold {fold+1}: Best Iter={model.best_iteration}, AUC={roc_auc_score(y_val, oof_probs[val_idx]):.4f}")
        
    avg_best_iter = int(np.mean(best_iters))
    print(f"[Model-XGB] Avg Best Iteration: {avg_best_iter}")
    
    # 2. 拟合 Platt Scaling (Logistic Regression on Logits)
    oof_logits = prob_to_logit(oof_probs)
    scaler = PlattScaler()
    scaler.fit(oof_logits, y_train)
    print(f"[Model-XGB] Platt Params: A={scaler.coefs[0]:.4f}, B={scaler.coefs[1]:.4f}")
    
    # 3. 全量训练
    print("[Model-XGB] Training Full Model...")
    final_rounds = int(avg_best_iter * 1.1)
    
    # 更新参数以移除 early_stopping (全量训练通常不需要)
    final_params = config['xgb_params'].copy()
    final_params['n_estimators'] = final_rounds
    
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train, verbose=False)
    
    # 4. 测试集预测
    test_probs_raw = final_model.predict_proba(X_test)[:, 1]
    test_logits = prob_to_logit(test_probs_raw)
    
    # 特征重要性
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': final_model,
        'test_scores_logit': test_logits,
        'platt_scaler': scaler,
        'importance': importances
    }

# ============================================================
# 5. 策略引擎 (Strategy Engine - Adaptive Center)
# ============================================================
def run_strategy_backtest(data_bundle, model_bundle, config):
    print("[Strategy] Running Backtest...")
    
    df = data_bundle['raw_test'].copy()
    
    # 1. 概率校准
    test_logits = model_bundle['test_scores_logit']
    scaler = model_bundle['platt_scaler']
    df['pred_prob'] = scaler.predict(test_logits)
    
    # 2. 自适应中枢 (Adaptive Center)
    # Pandas rolling 默认 right alignment (与 R 的 align='right' 一致)
    roll_window = config['prob_roll_window']
    slope = config['prob_slope']
    
    # 计算滚动中位数并 Lag 1 (shift 1)
    df['prob_baseline'] = df['pred_prob'].rolling(window=roll_window, min_periods=1).median().shift(1)
    
    # 填充初期空值
    df['prob_baseline'] = df['prob_baseline'].fillna(df['pred_prob'].median())
    
    # 信号生成
    df['raw_signal'] = 0.5 + (df['pred_prob'] - df['prob_baseline']) * slope
    
    # 3. 仓位合成
    # 限制范围 [0.2, 1.5]
    df['base_pos'] = df['raw_signal'].clip(lower=0.2, upper=1.5)
    
    # 波动率风控
    df['vol_20'] = df['market_forward_excess_returns'].rolling(20).std() * np.sqrt(252)
    current_vol = df['vol_20'].fillna(config['target_vol'])
    # 极小值保护
    current_vol = np.where(current_vol <= 0, config['target_vol'], current_vol)
    
    vol_scaler = config['target_vol'] / current_vol
    
    # 趋势滤网 (牛熊判断)
    trend_win = config['trend_window']
    # 计算价格指数
    df['price_index'] = (1 + df['market_forward_excess_returns']).cumprod()
    df['ma_trend'] = df['price_index'].rolling(trend_win, min_periods=1).mean()
    
    # is_bull: 价格 > 均线
    # 注意: R代码中这里并没有做 lag，如果是实时决策，最好做 shift(1)
    # 但为了保持与你 R 代码一致，这里不做 shift
    df['is_bull'] = df['price_index'] > df['ma_trend']
    trend_mult = np.where(df['is_bull'], 1.0, 0.6)
    
    # 最终仓位
    df['position'] = df['base_pos'] * vol_scaler * trend_mult
    df['position'] = df['position'].clip(upper=config['max_leverage']).fillna(0)
    
    print(f"  - Avg Position: {df['position'].mean():.2f}")
    
    # 4. 绩效计算 (含 Cost)
    # Cost = abs(diff(pos)) * rate
    pos_diff = df['position'].diff().fillna(df['position'].iloc[0]) # 假设初始从0仓位开始
    turnover_cost = pos_diff.abs() * config['transaction_cost']
    
    df['gross_return'] = df['position'] * df['market_forward_excess_returns']
    df['strategy_return'] = df['gross_return'] - turnover_cost
    
    print(f"  - Total Cost Impact: {turnover_cost.sum() * 100:.2f}%")
    
    # 统计指标
    def calc_metrics(ret_series):
        total_ret = (1 + ret_series).prod() - 1
        days = len(ret_series)
        ann_ret = (1 + total_ret) ** (252 / days) - 1
        ann_vol = ret_series.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        
        # Max DD
        cum_ret = (1 + ret_series).cumprod()
        drawdown = (cum_ret / cum_ret.cummax()) - 1
        max_dd = drawdown.min()
        
        return {
            'Ann_Return': ann_ret,
            'Sharpe': sharpe,
            'Max_DD': max_dd
        }
    
    metrics = calc_metrics(df['strategy_return'])
    mkt_metrics = calc_metrics(df['market_forward_excess_returns'])
    
    print("\n----------- XGB Strategy Performance -----------")
    print(f"Metrics     | Strategy | Market")
    print(f"Ann Return  | {metrics['Ann_Return']:.4f}   | {mkt_metrics['Ann_Return']:.4f}")
    print(f"Sharpe Ratio| {metrics['Sharpe']:.4f}   | {mkt_metrics['Sharpe']:.4f}")
    print(f"Max DD      | {metrics['Max_DD']:.4f}   | {mkt_metrics['Max_DD']:.4f}")
    
    # 绘图
    df['Cumulative Strategy'] = (1 + df['strategy_return']).cumprod()
    df['Cumulative Market'] = (1 + df['market_forward_excess_returns']).cumprod()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['date_id'], df['Cumulative Market'], label='Market', color='grey', alpha=0.6)
    plt.plot(df['date_id'], df['Cumulative Strategy'], label='XGB (Adaptive)', color='blue', linewidth=2)
    plt.title(f"XGBoost Strategy Performance (Sharpe: {metrics['Sharpe']:.2f})")
    plt.xlabel("Date ID")
    plt.ylabel("Net Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return df, metrics

# ============================================================
# 6. 主执行流 (Main Execution)
# ============================================================
if __name__ == "__main__":
    # 1. 数据处理
    data_bundle = process_data(CONFIG['input_file'], CONFIG)
    
    if data_bundle:
        # 2. 训练
        model_bundle = train_xgb_model(data_bundle, CONFIG)
        
        # 3. 回测
        res_df, metrics = run_strategy_backtest(data_bundle, model_bundle, CONFIG)