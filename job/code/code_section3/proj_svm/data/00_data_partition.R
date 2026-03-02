# ============================================================
# 脚本 00: 数据物理切分 (Data Partitioning)
# 目的: 构建"物理防火墙"，将训练集和测试集分离为不同文件，
#       彻底杜绝在训练和调参时偷看未来数据的可能性。
# ============================================================

rm(list = ls())
gc()
setwd("C:/Users/kagirasu/works_H/ml_HULL/job/code/code_section3/proj_svm")
# 1. 配置参数
# ------------------------------------------------------------
INPUT_FILE    <- "data/train.csv"
TRAIN_OUTPUT  <- "data/train_source.csv"  # 训练/验证专用 (可用于CV调参)
TEST_OUTPUT   <- "data/test_unseen.csv"   # 封存测试集 (只在最终跑分时读取)

SPLIT_DATE_ID <- 7000  # 切分点: <= 7000 为训练, > 7000 为测试

# 2. 读取原始数据
# ------------------------------------------------------------
cat(">>> [Partition] 正在读取原始数据:", INPUT_FILE, "...\n")

if (!file.exists(INPUT_FILE)) {
  stop("错误: 找不到原始数据文件，请检查路径！")
}

raw_data <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)

# 检查必要列
if (!"date_id" %in% names(raw_data)) {
  stop("错误: 数据中缺少 'date_id' 列，无法进行时序切分！")
}

# 3. 执行切分
# ------------------------------------------------------------
# 必须先按时间排序，确保逻辑正确
raw_data <- raw_data[order(raw_data$date_id), ]

# 切分
train_set <- subset(raw_data, date_id <= SPLIT_DATE_ID)
test_set  <- subset(raw_data, date_id >  SPLIT_DATE_ID)

# 4. 保存文件
# ------------------------------------------------------------
cat(">>> [Partition] 正在写入文件...\n")

write.csv(train_set, TRAIN_OUTPUT, row.names = FALSE)
cat("  - 已保存训练集:", TRAIN_OUTPUT, "\n")
cat("    行数:", nrow(train_set), "| 日期范围:", min(train_set$date_id), "~", max(train_set$date_id), "\n")

write.csv(test_set, TEST_OUTPUT, row.names = FALSE)
cat("  - 已保存封存集:", TEST_OUTPUT, "\n")
cat("    行数:", nrow(test_set),  " | 日期范围:", min(test_set$date_id), "~", max(test_set$date_id), "\n")

# 5. 验证完整性
# ------------------------------------------------------------
if (nrow(train_set) + nrow(test_set) == nrow(raw_data)) {
  cat("\n>>> [Success] 数据切分成功！总行数校验通过。\n")
} else {
  warning("\n>>> [Warning] 总行数校验失败，请检查是否有数据丢失！")
}