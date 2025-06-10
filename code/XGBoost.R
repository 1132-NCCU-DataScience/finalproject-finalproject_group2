## XGBoost

# load data
train <- read.csv("data/preparation/train.csv")
test <- read.csv("data/preparation/test.csv")

X_train <- train[, -c(1, 17, 18)]
y_train <- train$target

X_test <- test[, -c(1, 17, 18)]
y_test <- test$target

# train model
library(xgboost)

# prepare data
xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test))

# grid search parameter settings
fixed_params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    booster = "gbtree"
)

params_xgb <- expand.grid(
    max_depth = c(4, 6, 16, 32),
    eta = c(0.2, 0.1, 0.05),
    min_child_weight = c(1, 3),
    subsample = c(0.8, 1),
    colsample_bytree = c(0.8, 1)
)

# grid search
min_loss <- Inf
best_params_xgb <- list()
best_nrounds_xgb <- 0

for (i in 1: nrow(params_xgb)) {
    params <- c(fixed_params, as.list(params_xgb[i, ]))
    
    cv <- xgb.cv(
        params = params,
        data = xgb_train,
        nrounds = 1000,
        nfold = 5,
        early_stopping_rounds = 100,
        verbose = 0
    )
    
    loss <- mean(cv$evaluation_log$test_logloss_mean)
    
    if (loss < min_loss) {
        min_loss <- loss
        best_params_xgb <- params
        best_nrounds_xgb <- cv$best_iteration
    }
}

# construct the best model
xgb_model <- xgb.train(
    params = best_params_xgb,
    data = xgb_train,
    nrounds = best_nrounds_xgb,
    verbose = 0
)

# predict
pred <- predict(xgb_model, newdata = xgb_test)
pred_class <- ifelse(pred > 0.5, 1, 0)
mean(pred_class == y_test)  # acc: 0.5410759


# confusion matrix
library(dplyr)
df <- data.frame(
    predicted = pred_class,
    actual = y_test
)

confusion_df <- df %>%
    group_by(predicted, actual) %>%
    summarise(count = n(), .groups = "drop")

# plot
library(ggplot2)
ggplot(confusion_df, aes(x = predicted, y = actual, fill = count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = count), size = 6) +
    scale_fill_gradient(low = "#E0FFFF", high = "steelblue") +
    scale_y_reverse(breaks = c(0, 1)) +
    scale_x_continuous(breaks = c(0, 1)) +
    labs(
        title = "Confusion Matrix of XGBoost",
        x = "Predicted Label",
        y = "Actual Label"
    ) +
    theme_minimal() +
    theme(
        plot.title = element_text(hjust = 0.5, size = 20, face = "bold", margin = margin(b = 16, t = 16)),
        axis.text = element_text(size = 12),
        axis.title.x = element_text(size = 16, margin = margin(t = 16, b = 16)), 
        axis.title.y = element_text(size = 16, margin = margin(r = 16, l = 16)), 
        legend.box.margin = margin(l = 16, r = 16)
    )