## null model

# load data
train <- read.csv("data/preparation/train.csv")
test <- read.csv("data/preparation/test.csv")

X_train <- train[, -c(1, 17, 18)]

# predict function
null_pred <- function(target, var, app) {
    cuts <- unique(as.numeric(quantile(var, probs = seq(0, 1, 0.1), na.rm = T)))
    var_cat <- cut(var, cuts)
    app_cat <- cut(app, cuts)
    
    pPos <- sum(target == 1) / length(target)
    vTab <- table(as.factor(target), var_cat)
    pPosWv <- (vTab["1", ] + 1e-3 * pPos) / (colSums(vTab) + 1e-3)
    pred <- pPosWv[app_cat]
    pred[is.na(pred)] <- pPos
    
    return(pred)
}

# identify the best variable
best_acc <- 0
best_var <- ""

for(v in names(X_train)) {
    probs <- null_pred(train$target, train[[v]], test[[v]])
    pred_label <- ifelse(probs > 0.5, 1, 0)
    acc <- mean(pred_label == test$target)
    if(acc > best_acc) {
        best_acc <- acc
        best_var <- v
    }
}

cat(best_acc)  # 0.536997475
cat(best_var)  # rsi14