# ============================================================
# 模块 2: XGBoost 模型训练器 (适配 Sniper 接口)
# 文件路径: code_section4/RF_NEW/02_model_trainer_xgb.R
# ============================================================

library(xgboost)
library(dplyr)
library(caret) # 用于分层抽样 createFolds

train_xgb_model <- function(data_bundle, config) {
  
  cat("[Model-XGB] 启动 XGBoost 训练流程...\n")
  
  # 1. 数据准备 (转为 xgb.DMatrix)
  # XGBoost 只接受数值矩阵，process_data_rf 已经处理好了
  X_train_mat <- as.matrix(data_bundle$X_train)
  y_train_vec <- as.numeric(as.character(data_bundle$y_train)) # 0/1
  X_test_mat  <- as.matrix(data_bundle$X_test)
  
  dtrain_full <- xgb.DMatrix(data = X_train_mat, label = y_train_vec)
  dtest       <- xgb.DMatrix(data = X_test_mat)
  
  # 参数提取
  xp <- config$xgb_params
  xgb_params_core <- list(
    booster          = "gbtree",
    objective        = "binary:logistic", # 输出概率
    eval_metric      = "auc",             # 监控 AUC
    eta              = xp$eta,
    max_depth        = xp$max_depth,
    subsample        = xp$subsample,
    colsample_bytree = xp$colsample_bytree,
    min_child_weight = xp$min_child_weight,
    gamma            = xp$gamma,
    nthread          = parallel::detectCores() - 1, # 自动并行
    seed             = config$seed
  )
  
  # ============================================================
  # 2. 内部 CV 获取 OOF (Out-of-Fold) 概率
  # 这是 Sniper 策略"冷启动"的关键
  # ============================================================
  cat("[Model-XGB] 执行 5-Fold CV 以获取历史概率 (用于 Sniper 锚定)...\n")
  
  k_folds <- 5

  # [关键修正 1] 锁定 CV 切分的随机性
  # 必须在 createFolds 之前设种子，否则每次切分的数据行都不一样
  set.seed(config$seed)
  # 使用 caret 创建分层折叠，保证正负样本比例一致
  folds <- caret::createFolds(y_train_vec, k = k_folds, list = TRUE)
  
  cv_probs_oof <- numeric(length(y_train_vec)) # 存储 OOF 预测
  best_nrounds_list <- numeric(k_folds)
  
  for(k in 1:k_folds) {
    val_idx <- folds[[k]]
    train_idx <- setdiff(seq_along(y_train_vec), val_idx)
    
    dtr <- xgb.DMatrix(data = X_train_mat[train_idx, ], label = y_train_vec[train_idx])
    dval <- xgb.DMatrix(data = X_train_mat[val_idx, ], label = y_train_vec[val_idx])
    
    # 带早停的训练
    watchlist <- list(train = dtr, eval = dval)
    bst_cv <- xgb.train(
      params          = xgb_params_core,
      data            = dtr,
      nrounds         = xp$nrounds,
      watchlist       = watchlist,
      early_stopping_rounds = xp$early_stopping,
      verbose         = 0 # 静默模式
    )
    
    # 记录最佳轮数
    best_nrounds_list[k] <- bst_cv$best_iteration
    
    # 预测验证集 (得到的是概率)
    preds <- predict(bst_cv, dval)
    cv_probs_oof[val_idx] <- preds
    
    cat(sprintf("  - Fold %d: Best Iter=%d, AUC=%.4f\n", k, bst_cv$best_iteration, bst_cv$best_score))
  }
  
  avg_best_nrounds <- floor(mean(best_nrounds_list))
  cat(sprintf("[Model-XGB] CV 完成. 平均最佳轮数: %d\n", avg_best_nrounds))
  
  # ============================================================
  # 3. 概率校准 (Platt Scaling)
  # 虽然 XGB 输出本身就是概率，但为了与 RF 接口保持一致并进一步校准
  # 我们先将其转回 Logits，再拟合 Platt
  # ============================================================
  eps <- 1e-6
  cv_probs_clipped <- pmax(eps, pmin(1 - eps, cv_probs_oof))
  cv_logits <- log(cv_probs_clipped / (1 - cv_probs_clipped)) # 反向变换
  
  # 拟合 Platt 参数 (A, B)
  platt_params <- algo_platt_fit(cv_logits, y_train_vec)
  
  # 生成最终校准后的训练集概率 (传给 Strategy 做历史分位数)
  train_cv_probs_calibrated <- algo_platt_predict(cv_logits, platt_params)
  
  # ============================================================
  # 4. 全量训练
  # ============================================================
  cat("[Model-XGB] 使用最佳轮数进行全量训练...\n")
  
  # 全量训练通常稍微增加一点轮数(因为数据变多了)，或者直接用平均值
  final_nrounds <- floor(avg_best_nrounds * 1.1) 
  
  model_full <- xgb.train(
    params    = xgb_params_core,
    data      = dtrain_full,
    nrounds   = final_nrounds,
    verbose   = 0
  )
  
  # ============================================================
  # 5. 预测测试集 & 特征重要性
  # ============================================================
  cat("[Model-XGB] 生成测试集预测...\n")
  
  test_probs_raw <- predict(model_full, dtest)
  
  # 转 Logit (为了适配 Strategy 里的 algo_platt_predict)
  test_probs_clipped <- pmax(eps, pmin(1 - eps, test_probs_raw))
  test_scores_logit  <- log(test_probs_clipped / (1 - test_probs_clipped))
  
  # 特征重要性
  importance_matrix <- xgb.importance(feature_names = colnames(X_train_mat), model = model_full)
  # 格式对齐
  imp_df <- data.frame(
    Feature = importance_matrix$Feature,
    Frequency = importance_matrix$Gain # XGB中 Gain 通常比 Frequency 更重要
  )
  
  write.csv(imp_df, file.path(config$output_dir, "xgb_importance.csv"), row.names = FALSE)
  
  # 返回符合 Bundle 规范的列表
  return(list(
    model              = model_full,
    test_scores        = test_scores_logit,       # Logits 供 Calibration 用
    platt_params       = platt_params,            # 校准参数
    train_cv_probs     = train_cv_probs_calibrated, # [关键] Sniper 历史数据
    feature_importance = imp_df,
    model_type         = "xgboost"
  ))
}