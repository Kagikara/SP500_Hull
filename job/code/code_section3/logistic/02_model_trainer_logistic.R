# ============================================================
# 模块 2: Logistic 模型训练器 (修复: 增加 Platt 校准)
# 文件路径: logistic/02_model_trainer_logistic.R
# ============================================================

library(dplyr)
# 需要 algorithm_handmade.r 中的 algo_platt_fit

train_logistic_model <- function(data_bundle, config) {
  
  cat("[Model-Logit] 开始训练 Logistic 回归模型...\n")
  
  train_df <- data_bundle$train
  test_df  <- data_bundle$test
  
  # 确定公式
  formula_str <- if (config$use_interaction) "y ~ P_std * V_std" else "y ~ P_std + V_std"
  
  # ============================================================
  # 1. [关键修复] Internal CV 拟合 Platt Scaling 参数
  # Logistic 原始概率可能因正则化或数据分布偏移而不准
  # ============================================================
  cat("[Model-Logit] 执行 3-Fold CV 以校准概率 (Platt Scaling)...\n")
  
  k_folds <- 3
  folds <- cut(seq(1, nrow(train_df)), breaks = k_folds, labels = FALSE)
  cv_logits <- numeric(nrow(train_df))
  
  for(k in 1:k_folds) {
    idx_val <- which(folds == k)
    idx_tr  <- which(folds != k)
    
    # 训练 Fold 模型
    fold_model <- glm(as.formula(formula_str), 
                      data = train_df[idx_tr, ], 
                      family = binomial(link = "logit"))
    
    # 预测 Logits (link)
    cv_logits[idx_val] <- predict(fold_model, newdata = train_df[idx_val, ], type = "link")
  }
  
  # 拟合参数 A, B
  platt_params <- algo_platt_fit(cv_logits, as.numeric(train_df$y))
  cat(sprintf("  - 校准参数: A=%.4f, B=%.4f\n", platt_params$A, platt_params$B))

  # ============================================================
  # 2. 训练全量模型
  # ============================================================
  cat("[Model-Logit] 训练全量模型...\n")
  
  final_model <- glm(as.formula(formula_str), 
                     data = train_df, 
                     family = binomial(link = "logit"))
  
  print(summary(final_model))
  
  # ============================================================
  # 3. 生成测试集预测 (输出 Logits)
  # ============================================================
  cat("[Model-Logit] 生成测试集预测 (Logits)...\n")
  
  # 注意：这里我们输出 type="link" (即 logits/scores)，
  # 交给策略引擎统一使用 platt_params 转为概率
  test_logits <- predict(final_model, newdata = test_df, type = "link")
  
  return(list(
    model = final_model,
    test_scores = test_logits,   # 输出 Logits
    platt_params = platt_params, # 输出校准参数
    formula = formula_str
  ))
}