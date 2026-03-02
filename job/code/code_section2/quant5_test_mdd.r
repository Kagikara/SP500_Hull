library(tidyverse)

data_P <- read.csv("train.csv", stringsAsFactors = FALSE)

p_colindex <- c(58:70)
P_star_signal <- rowMeans(data_P[, p_colindex], na.rm = TRUE)

R_m_return <- data_P$market_forward_excess_returns

P_star_filtered <- P_star_signal[-(1:1006)]
R_m_filtered <- R_m_return[-(1:1006)]

n_target <- floor(length(P_star_filtered) * 0.20)
sorted_indices <- order(P_star_filtered)
label <- rep(FALSE, length(P_star_filtered))
label[sorted_indices[1:n_target]] <- TRUE

R_Q1 <- ifelse(label, R_m_filtered, 0)
R_P <- R_m_filtered

# 构建净值曲线
V_Q1 <- cumprod(1 + R_Q1)
V_P <- cumprod(1 + R_P)

x_axis <- 1:length(R_P)

# 绘制 V_P
plot(x = x_axis, y = V_P, type = 'l', col = 'blue', 
     main = "策略净值曲线对比",
     xlab = "交易日 (自 date_id 1007 起)", 
     ylab = "净值 (初始 $1)")

# V_Q1
lines(x = x_axis, y = V_Q1, col = 'red')

# 图例
legend("topleft", 
       legend = c("V_P", "V_Q1"), 
       col = c("blue", "red"), 
       lty = 1)