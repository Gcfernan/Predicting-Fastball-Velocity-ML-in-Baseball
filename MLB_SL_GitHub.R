# Super Learner Ensemble Model for Predicting Fastball Velocity
# ---------------------------------------------------------------

# --------------------
# Load Required Packages
# --------------------
packages <- c("SuperLearner", "caret", "ranger", "xgboost", "glmnet", "earth", "kernlab", 
              "naniar", "ggplot2", "ggrepel", "dplyr", "reshape2", "stringr")
lapply(packages, function(pkg) if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg))
lapply(packages, library, character.only = TRUE)

# ----------------------
# Sample Size Calculation
# ----------------------
# Based on expected R² of 0.30, 15 parameters, outcome mean of 80.8 and SD of 7.6
pmsampsize::pmsampsize(type = "c", rsquared = 0.30, parameters = 15, intercept = 80.8, sd = 7.6)

# ----------------------
# Load and Preprocess Dataset 
# ----------------------
data <- read.csv("data.csv")
data <- data %>% filter(!is.na(Lead_Knee_Ang_Vel_Max_PreBR))

# Extract conference from ID 
data$conference <- str_extract(data$PitcherID, "ACC|SEC")


data_combined <- data %>%
  select(-PitchID) %>%
  group_by(PitcherID, across(where(is.factor) | where(is.character))) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE), .groups = "drop")

X <- data_combined %>% select(-RelSpeed, -PitcherID, -conference)
y <- data_combined$RelSpeed
X$Handedness <- as.factor(X$Handedness)

# -------------------------
# Define and Register Tuned Learners
# -------------------------
create_tuned_learners <- function() {
  list(
    SL.ranger.tuned = function(...) SL.ranger(..., num.trees = 500, mtry = 3, min.node.size = 5),
    SL.xgboost.tuned = function(...) SL.xgboost(..., nrounds = 100, max_depth = 3, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8),
    SL.glmnet.tuned = function(...) SL.glmnet(..., alpha = 0.5),
    SL.earth.tuned = function(...) SL.earth(..., degree = 2, nk = min(20, ncol(X)+2)),
    SL.ksvm.tuned = function(...) SL.ksvm(..., kernel = "rbfdot", C = 1)
  )
}
tuned_learners <- create_tuned_learners()
for (name in names(tuned_learners)) {
  environment(tuned_learners[[name]]) <- asNamespace("SuperLearner")
  assign(name, tuned_learners[[name]], envir = .GlobalEnv)
}

sl_lib <- c("SL.ranger.tuned", "SL.xgboost.tuned", "SL.glmnet.tuned", "SL.earth.tuned", "SL.ksvm.tuned", "SL.mean")

# -------------------------
# Cross-Validation with Summary Metrics + 95% CI
# -------------------------
k_folds <- 10
set.seed(456)
folds <- caret::createFolds(y, k = k_folds, returnTrain = FALSE)
metrics <- data.frame(Fold = 1:k_folds, MSE = NA, RMSE = NA, MAE = NA, R2 = NA)

for (i in 1:k_folds) {
  idx_test <- folds[[i]]
  x_train <- X[-idx_test, ]; y_train <- y[-idx_test]
  x_test <- X[idx_test, ]; y_test <- y[idx_test]
  model <- SuperLearner(Y = y_train, X = x_train, family = gaussian(), SL.library = sl_lib, cvControl = list(V = 5))
  pred <- predict(model, newdata = x_test)$pred
  metrics[i, 2:5] <- c(
    mean((y_test - pred)^2),
    sqrt(mean((y_test - pred)^2)),
    mean(abs(y_test - pred)),
    1 - sum((y_test - pred)^2) / sum((y_test - mean(y_test))^2)
  )
}

confidence_interval <- function(mean, sd, n){
  error <- 1.96 * (sd / sqrt(n))
  c(lower = mean - error, upper = mean + error)
}

summary_metrics <- metrics %>% summarise(across(MSE:R2, list(mean = mean, sd = sd)))
ci_table <- data.frame(
  Metric = names(summary_metrics)[seq(1, 8, 2)],
  Mean = unlist(summary_metrics[1, seq(1, 8, 2)]),
  Lower = mapply(confidence_interval, unlist(summary_metrics[1, seq(1, 8, 2)]), unlist(summary_metrics[1, seq(2, 8, 2)]), MoreArgs = list(n = k_folds))[1,],
  Upper = mapply(confidence_interval, unlist(summary_metrics[1, seq(1, 8, 2)]), unlist(summary_metrics[1, seq(2, 8, 2)]), MoreArgs = list(n = k_folds))[2,]
)
print("Cross-Validation Performance with 95% CI:")
print(ci_table)

# --------------------------
# Final Model + Prediction
# --------------------------
final_model <- SuperLearner(Y = y, X = X, family = gaussian(), SL.library = sl_lib, cvControl = list(V = 10))
final_preds <- predict(final_model, newdata = X)$pred

mse <- mean((y - final_preds)^2)
rmse <- sqrt(mse)
mae <- mean(abs(y - final_preds))
r2 <- 1 - sum((y - final_preds)^2) / sum((y - mean(y))^2)
cat(sprintf("Final Model Performance:\nRMSE = %.3f | MAE = %.3f | R² = %.3f\n", rmse, mae, r2))

# --------------------------
# Variable Importance (Weighted by Ensemble)
# --------------------------
get_superlearner_importance <- function(sl_model, X, Y) {
  weights <- sl_model$coef
  var_names <- colnames(X)
  imp_matrix <- matrix(0, ncol = length(var_names), nrow = 0)
  colnames(imp_matrix) <- var_names
  
  for (alg in names(weights)) {
    if (weights[alg] <= 0 || alg == "SL.mean") next
    fit <- sl_model$fitLibrary[[alg]]$object
    imp <- rep(0, ncol(X)); names(imp) <- var_names
    
    if (grepl("ranger", alg) && !is.null(fit$variable.importance)) {
      norm_imp <- fit$variable.importance / sum(fit$variable.importance)
      imp[names(norm_imp)] <- norm_imp
    } else if (grepl("xgboost", alg)) {
      imp_tbl <- xgb.importance(model = fit)
      if (nrow(imp_tbl) > 0) {
        norm_imp <- imp_tbl$Gain / sum(imp_tbl$Gain)
        imp[imp_tbl$Feature] <- norm_imp
      }
    } else if (grepl("glmnet", alg)) {
      coef_vals <- coef(fit, s = "lambda.min")[-1]
      norm_coef <- abs(as.numeric(coef_vals)) / sum(abs(coef_vals))
      imp[names(coef_vals)] <- norm_coef
    }
    imp_matrix <- rbind(imp_matrix, imp * weights[alg])
  }
  importance <- colSums(imp_matrix)
  importance <- 100 * importance / sum(importance)
  return(sort(importance, decreasing = TRUE))
}

importance <- get_superlearner_importance(final_model, X, y)
print("Variable Importance (% Weighted):")
print(round(importance, 2))

# --------------------------
# Internal-External Validation (Bootstrap by Conference)
# --------------------------
perform_bootstrap_conference_split <- function(data, train_conferences, test_conferences, sl_lib, n_bootstrap = 100, seed = 42) {
  set.seed(seed)
  train_data <- data %>% filter(conference %in% train_conferences)
  test_data <- data %>% filter(conference %in% test_conferences)
  train_x <- train_data %>% select(-RelSpeed, -conference, -PitcherID)
  train_y <- train_data$RelSpeed
  test_x <- test_data %>% select(-RelSpeed, -conference, -PitcherID)
  test_y <- test_data$RelSpeed
  
  metrics <- replicate(n_bootstrap, {
    idx <- sample(1:nrow(train_data), replace = TRUE)
    boot_x <- train_x[idx, ]; boot_y <- train_y[idx]
    model <- SuperLearner(Y = boot_y, X = boot_x, family = gaussian(), SL.library = sl_lib)
    preds <- predict(model, newdata = test_x)$pred
    c(RMSE = sqrt(mean((test_y - preds)^2)),
      MAE = mean(abs(test_y - preds)),
      R2 = 1 - sum((test_y - preds)^2) / sum((test_y - mean(test_y))^2))
  })
  metrics <- as.data.frame(t(metrics))
  results <- data.frame(
    Metric = names(metrics),
    Mean = sapply(metrics, mean),
    Lower = sapply(metrics, function(x) quantile(x, 0.025)),
    Upper = sapply(metrics, function(x) quantile(x, 0.975))
  )
  return(results)
}

# Example call:
bootstrap_ACC <- perform_bootstrap_conference_split(data_combined, "SEC", "ACC", sl_lib, n_bootstrap = 20)
bootstrap_SEC <- perform_bootstrap_conference_split(data_combined, "ACC", "SEC", sl_lib, n_bootstrap = 20)
print("Bootstrap SEC to ACC:")
print(bootstrap_ACC)
print("Bootstrap ACC to SEC:")
print(bootstrap_SEC)

# --------------------------
# Calibration Plot 
# --------------------------
plot_df <- data.frame(Actual = y, Predicted = final_preds)
cal_model <- lm(Actual ~ Predicted, data = plot_df)
cal_ci <- confint(cal_model)
cat(sprintf("Calibration: Intercept = %.2f (%.2f–%.2f), Slope = %.2f (%.2f–%.2f)\n",
            coef(cal_model)[1], cal_ci[1,1], cal_ci[1,2],
            coef(cal_model)[2], cal_ci[2,1], cal_ci[2,2]))

p <- ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Super Learner Calibration", x = "Actual Velocity", y = "Predicted Velocity") +
  theme_minimal()
print(p)

# --------------------------
# End of Script
# --------------------------
