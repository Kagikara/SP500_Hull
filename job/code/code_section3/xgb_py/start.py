import os
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import joblib
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# ============================================================
# 1. 策略配置 (保持与之前一致)
# ============================================================
CONFIG = {
    # 路径要根据Kaggle实际情况调整
    'input_file': '/kaggle/input/hull-tactical-market-prediction/train.csv', 
    'seed': 42,
    
    # 策略参数
    'prob_roll_window': 60,   # 自适应窗口
    'prob_slope': 10.0,       # 信号放大倍数
    'max_leverage': 2.0,      # 杠杆硬顶
    
    # XGB 参数
    'xgb_params': {
        'n_estimators': 1500,
        'learning_rate': 0.01,
        'max_depth': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'objective': 'binary:logistic',
        'n_jobs': -1,
        'random_state': 42,
        'tree_method': 'hist'
    }
}

# ============================================================
# 2. 辅助类: 策略预测器 (Stateful Predictor)
# ============================================================
class PlattScaler:
    def __init__(self):
        self.lr = LogisticRegression(C=1e9, solver='lbfgs')
        
    def fit(self, logits, labels):
        self.lr.fit(logits.reshape(-1, 1), labels)
        
    def predict(self, logits):
        return self.lr.predict_proba(logits.reshape(-1, 1))[:, 1]

def prob_to_logit(probs, eps=1e-6):
    probs = np.clip(probs, eps, 1 - eps)
    return np.log(probs / (1 - probs))

class StrategyPredictor:
    def __init__(self, config):
        self.cfg = config
        self.model = None
        self.scaler = None
        self.features = []
        
        # [核心] 状态保持器: 存储历史概率
        self.prob_history = deque(maxlen=config['prob_roll_window'])
        
        # 训练并初始化状态
        self._train_and_init()

    def _train_and_init(self):
        print(">>> [Init] Loading Data & Training Model...")
        try:
            df = pd.read_csv(self.cfg['input_file'])
        except:
            print("Warning: Input file not found. Skipping training (for dry-run).")
            return

        # 1. 基础处理
        df = df[df['date_id'] >= 1006].sort_values('date_id').reset_index(drop=True)
        df['target'] = (df['market_forward_excess_returns'] >= 0).astype(int)
        
        # 2. 特征定义
        exclude = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'target', 'y_target']
        self.features = [c for c in df.columns if c not in exclude]
        
        # 简单填充 (XGB能处理NA，但为了安全先填0)
        X = df[self.features].fillna(0)
        y = df['target']
        
        # 3. 训练 XGB (使用全部数据)
        self.model = xgb.XGBClassifier(**self.cfg['xgb_params'])
        self.model.fit(X, y, verbose=False)
        
        # 4. 拟合 Platt Scaler (简化版: 直接用 OOF 或 训练集预测)
        # 为了速度，这里简单使用全量预测来拟合 Scaler (严谨比赛建议用CV)
        raw_probs = self.model.predict_proba(X)[:, 1]
        raw_logits = prob_to_logit(raw_probs)
        
        self.scaler = PlattScaler()
        self.scaler.fit(raw_logits, y)
        calibrated_probs = self.scaler.predict(raw_logits)
        
        # 5. [关键] 初始化历史记忆 (Cold Start)
        # 将训练集最后 60 天的"校准后概率"填入队列
        # 这样第 1 次预测时，我们就能算出中位数基准
        last_probs = calibrated_probs[-self.cfg['prob_roll_window']:]
        self.prob_history.extend(last_probs)
        
        print(f">>> [Init] Done. History Buffer Size: {len(self.prob_history)}")

    def predict_one(self, test_df_polars):
        """
        处理 API 传来的单行数据
        """
        # 1. 格式转换 Polars -> Pandas
        current_df = test_df_polars.to_pandas()
        
        # 2. 特征对齐
        # 确保列存在，缺失补0
        for f in self.features:
            if f not in current_df.columns:
                current_df[f] = 0
        
        X_curr = current_df[self.features].fillna(0)
        
        # 3. 模型推理
        if self.model is None: return 0.0 # 防御代码
        
        raw_prob = self.model.predict_proba(X_curr)[:, 1][0]
        raw_logit = prob_to_logit(np.array([raw_prob]))
        curr_prob = self.scaler.predict(raw_logit)[0]
        
        # 4. [策略逻辑] 自适应中枢
        # A. 计算历史中位数 (Baseline)
        if len(self.prob_history) > 0:
            baseline = np.median(self.prob_history)
        else:
            baseline = 0.5 # 兜底
            
        # B. 更新历史 (把当前的加入队列，最老的会被自动挤出)
        self.prob_history.append(curr_prob)
        
        # C. 信号计算: 0.5 + (Prob - Baseline) * Slope
        slope = self.cfg['prob_slope']
        raw_signal = 0.5 + (curr_prob - baseline) * slope
        
        # 5. 仓位限制
        # 限制在 [0.2, 1.5] 并应用硬顶
        position = np.clip(raw_signal, 0.2, 1.5)
        position = min(position, self.cfg['max_leverage'])
        
        # 返回 float 类型
        return float(position)

# ============================================================
# 3. 全局初始化 (在 API 启动前运行一次)
# ============================================================
# 创建全局预测器实例
# 这行代码会在 Submit 后，环境启动时运行
GLOBAL_PREDICTOR = StrategyPredictor(CONFIG)

# ============================================================
# 4. 定义 Kaggle API 接口
# ============================================================
import kaggle_evaluation.default_inference_server

def predict(test: pl.DataFrame) -> float:
    """
    API 回调函数。每次由 Gateway 调用。
    """
    try:
        # 调用全局实例进行预测
        return GLOBAL_PREDICTOR.predict_one(test)
    except Exception as e:
        # 容错处理：如果出错，返回空仓，防止报错导致提交失败
        # print(f"Error: {e}") 
        return 0.0

# 启动服务
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # 本地调试模式 (会读取 input 里的数据模拟 API)
    inference_server.run_local_gateway(
        (CONFIG['input_file'].replace('train.csv', ''),)
    )