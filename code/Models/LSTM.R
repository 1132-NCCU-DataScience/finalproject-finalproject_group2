library(tidyverse)
library(keras)
library(abind)

stock <- read.csv("data/preparation2_standardization.csv")

# train-test split
set.seed(1000)
train <- stock %>% group_by(target) %>% sample_frac(0.8)
test <- anti_join(stock, train)

X_train <- train[, -c(1, 2)]
y_train <- train$target

X_test <- test[, -c(1, 2)]
y_test <- test$target

# 建立訓練/測試資料集(每個樣本是連續 time_step 天)
create_dataset <- function(data, time_step) {
    n <- nrow(data)
    
    X_list <- list()
    for(i in 1:(n - time_step)) {
        sample_window <- data[i: (i + time_step - 1), ]
        X_list[[i]] <- as.matrix(sample_window)
    }
    
    X_array <- abind(X_list, along = 0)
    return(X_array)
}

time_step <- 14
n_features <- ncol(X_train)

train_data <- create_dataset(X_train, time_step)
train_label <- y_train[(time_step + 1): length(y_train)]

test_data <- create_dataset(X_test, time_step)
test_label <- y_test[(time_step + 1): length(y_test)]

# 測試不同的超參數
best_acc <- 0
best_val_acc <- 0
best_params <- list()

params <- expand.grid(
    lr = c(7e-3, 5e-3),   # learning rate
    bs = c(32, 64),   # batch size
    ep = c(30, 50),   # epochs
    drop = c(0.3, 0.4)  # dropout rate
)

for(i in 1: nrow(params)) {
    model <- keras_model_sequential() %>%
        layer_lstm(units = 32, input_shape = c(time_step, n_features), return_sequences = T) %>%
        layer_dropout(rate = params$drop[i]) %>%
        layer_lstm(units = 64, return_sequences = F) %>%
        layer_dropout(rate = params$drop[i]) %>%
        layer_dense(units = 32, activation = "relu") %>%
        layer_dense(units = 16, activation = "relu") %>%
        layer_dense(units = 1)
    
    model %>% compile(
        loss = "binary_crossentropy",
        optimizer = optimizer_adam(learning_rate = params$lr[i]),
        metrics = c("accuracy")
    )
    
    history <- model %>% fit(
        x = train_data,
        y = train_label,
        epochs = params$ep[i],
        batch_size = params$bs[i],
        validation_split = 0.2
    )
    
    val_acc <- max(history$metrics$val_accuracy)
    if(val_acc > best_val_acc) {
        best_val_acc <- val_acc
        best_params <- params[i, ]
        best_acc <- max(history$metrics$accuracy)
    }
}

print(best_params)

# 建構最佳模型
model <- keras_model_sequential() %>%
    layer_lstm(units = 32, input_shape = c(time_step, n_features), return_sequences = T) %>%
    layer_dropout(rate = 0.3) %>%
    layer_lstm(units = 64, return_sequences = F) %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1)

model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = 5e-3),
    metrics = c("accuracy")
)

history <- model %>% fit(
    x = train_data,
    y = train_label,
    epochs = 30,
    batch_size = 32, 
    validation_data = list(test_data, test_label),
    callbacks = list(
        callback_model_checkpoint(
            filepath = "best_model.h5",
            monitor = "val_loss",
            save_best_only = TRUE,
            save_weights_only = FALSE
        )
    )
)

# 載入模型
# model <- load_model_hdf5("best_model.h5")
