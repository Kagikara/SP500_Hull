library(PeerPerformance) # 包含Ledoit-Wolf 检验的 sharpeTesting 函数

data_P <- read.csv("train.csv", stringsAsFactors = FALSE)
colindex <- c(58:70)
P_star <- rowMeans(data_P[, colindex], na.rm = TRUE)
P_star <- P_star[-(1:1006)] # 只从第1006条数据开始，因为前面的数据不含有该特征

n_target <- floor(length(P_star) * 0.20)  # 目标数量
sorted_indices <- order(P_star)  # 排序后的索引
label <- rep(FALSE, length(P_star))
label[sorted_indices[1:n_target]] <- TRUE  # 标记Q1组

R_Q1 <- ifelse(label, P_star, 0)
R_P <- P_star

# type = 2 表示使用studentized bootstrap 
# bBoot = 0 表示让函数自动选择最佳块长度
control_params <- list(type = 2, nBoot = 4999, bBoot = 0)

# 执行检验
test_result <- sharpeTesting(R_Q1, R_P, control = control_params)

# 查看 p-value
test_result