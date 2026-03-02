data_P <- read.csv("train.csv", stringsAsFactors = FALSE)

p_colindex <- c(58:70)
P_star_signal <- rowMeans(data_P[, p_colindex], na.rm = TRUE)[-(1:1006)]
v_colindex <- c(83:95)
V_star_signal <- rowMeans(data_P[, v_colindex], na.rm = TRUE)[-(1:1006)]
training_data <- as.data.frame(cbind(P_star_signal, V_star_signal))
y_zero_one <- ifelse(data_P[98] >= 0, 1, 0)[-(1:1006)]

logit_model <- glm(y_zero_one ~ ., 
                   data = training_data, 
                   family = binomial(link = "logit"))

summary(logit_model)
cor(training_data$P_star_signal, training_data$V_star_signal)

logit_model_interaction <- glm(y_zero_one ~ P_star_signal + V_star_signal + P_star_signal:V_star_signal, 
                               data = training_data, 
                               family = binomial(link = "logit"))
summary(logit_model_interaction)
