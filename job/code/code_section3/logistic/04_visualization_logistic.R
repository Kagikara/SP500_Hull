# ============================================================
# 模块 4: Logistic 可视化分析工具
# 文件路径: logistic/04_visualization_logistic.R
# ============================================================

library(ggplot2)
library(dplyr)
library(gridExtra)

# 辅助函数: Sigmoid (用于手动复现 Platt 预测)
sigmoid <- function(x) { 1 / (1 + exp(-x)) }

visualize_logistic_results <- function(data_bundle, model_results, config) {
  
  cat("[Viz-Logit] 开始生成可视化图表...\n")
  
  # 提取必要对象
  model        <- model_results$model
  platt_params <- model_results$platt_params
  train_df     <- data_bundle$train
  test_df      <- data_bundle$test
  
  # ============================================================
  # 图 1: 决策边界热力图 (Decision Boundary Heatmap)
  # 目的: 展示 P_std 和 V_std 如何共同决定上涨概率 (含交互项效果)
  # ============================================================
  
  # 1. 创建网格 (Grid) 覆盖数据的取值范围
  p_range <- range(train_df$P_std)
  v_range <- range(train_df$V_std)
  
  # 适当外扩一点范围，让图更好看
  grid_data <- expand.grid(
    P_std = seq(p_range[1]*1.1, p_range[2]*1.1, length.out = 100),
    V_std = seq(v_range[1]*1.1, v_range[2]*1.1, length.out = 100)
  )
  
  # 2. 预测网格中每个点的 Logits
  # 注意: 即使模型包含交互项，predict 函数会自动根据 P_std 和 V_std 计算 P*V
  grid_data$logits <- predict(model, newdata = grid_data, type = "link")
  
  # 3. 应用 Platt Scaling 转为概率
  # Prob = 1 / (1 + exp(-(A * logit + B)))
  grid_data$prob <- sigmoid(platt_params$A * grid_data$logits + platt_params$B)
  
  # 4. 绘图
  p1 <- ggplot(grid_data, aes(x = P_std, y = V_std)) +
    # 背景热力图
    geom_tile(aes(fill = prob)) +
    # 颜色映射: 0(蓝)-0.5(白)-1(红)
    scale_fill_gradient2(low = "#2166ac", mid = "#f7f7f7", high = "#b2182b", 
                         midpoint = 0.5, limits = c(0, 1), name = "上涨概率") +
    # 叠加 0.5 决策边界线
    geom_contour(aes(z = prob), breaks = 0.5, color = "black", size = 0.8, linetype = "dashed") +
    # (可选) 叠加少量测试集真实散点，观察模型是否包住了正例
    geom_point(data = test_df %>% sample_n(min(500, nrow(test_df))), 
               aes(color = as.factor(y)), alpha = 0.4, size = 1.5, shape = 16) +
    scale_color_manual(values = c("0"="blue", "1"="red"), name = "真实标签") +
    labs(title = "Logistic 决策边界与交互效应",
         subtitle = "虚线为 50% 概率分界线; 背景越红代表预测上涨概率越高",
         x = "价格因子 (P_std)", y = "成交量因子 (V_std)") +
    theme_minimal() +
    theme(legend.position = "right",
    panel.grid = element_blank(),       # <--- 新增: 去除主网格
    panel.grid.major = element_blank(), # <--- 确保去除大网格
    panel.grid.minor = element_blank()
    )  # <--- 确保去除小网格)

  # ============================================================
  # 图 2: 校准曲线 (Calibration Curve)
  # 目的: 验证模型预测的 0.8 概率是否真的对应 80% 的上涨比例
  # ============================================================
  
  # 1. 获取测试集的最终预测概率
  test_logits <- predict(model, newdata = test_df, type = "link")
  test_probs  <- sigmoid(platt_params$A * test_logits + platt_params$B)
  
  cal_df <- data.frame(
    prob = test_probs,
    y    = test_df$y
  )
  
  # 2. 分箱 (Binning) - 将概率分为 10 组
  num_bins <- 10
  cal_df$bin <- cut(cal_df$prob, breaks = seq(0, 1, length.out = num_bins + 1), include.lowest = TRUE)
  
  # 3. 计算每个箱的统计量
  calibration_data <- cal_df %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(prob),           # 预测概率均值
      fraction_pos = mean(y),           # 真实正例比例
      n = n(),                          # 样本数
      se = sqrt(fraction_pos * (1 - fraction_pos) / n) # 标准误 (用于误差棒)
    ) %>%
    na.omit() # 去除空箱
  
  # 4. 绘图
  p2 <- ggplot(calibration_data, aes(x = mean_pred, y = fraction_pos)) +
    # 完美校准线 (y=x)
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
    # 误差棒
    geom_errorbar(aes(ymin = fraction_pos - se, ymax = fraction_pos + se), 
                  width = 0.02, color = "gray60") +
    # 数据点和连线
    geom_line(color = "steelblue", size = 1) +
    geom_point(aes(size = n), color = "steelblue") +
    # 标尺限制
    xlim(0, 1) + ylim(0, 1) +
    labs(title = "Platt 校准曲线 (Reliability Diagram)",
         subtitle = "点越靠近虚线对角线，说明概率越可信",
         x = "模型预测概率", y = "实际发生比例",
         size = "样本量") +
    theme_minimal() +
    theme(legend.position = "bottom")

  # ============================================================
  # 保存与输出
  # ============================================================
  
  # 组合图形 (左边热力图，右边校准曲线)
  combined_plot <- arrangeGrob(p1, p2, ncol = 2, 
                               top = textGrob("Logistic 模型诊断报告", 
                                              gp = gpar(fontsize = 16, fontface = "bold")))
  
  outfile <- file.path(config$output_dir, "logistic_model_diagnosis.png")
  ggsave(outfile, combined_plot, width = 16, height = 7, dpi = 300)
  
  cat(sprintf("可视化图表已保存: %s\n", outfile))
}