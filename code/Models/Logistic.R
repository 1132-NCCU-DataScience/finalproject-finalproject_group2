## Logistic

# load data
train <- read.csv("data/preparation/train.csv")
test <- read.csv("data/preparation/test.csv")

X_train <- train[, -c(1, 17, 18)]
y_train <- train$target

X_test <- test[, -c(1, 17, 18)]
y_test <- test$target

# train model
log_train <- cbind(X_train, label = as.factor(y_train))
logit_model <- glm(label ~ ., data = log_train, family = binomial(link = "logit"))

# predict
pred_log <- predict(logit_model, newdata = X_test, type = "response")
pred_class <- ifelse(pred_log > 0.5, 1, 0)
mean(pred_class == y_test)  # acc: 0.5641872


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
        title = "Confusion Matrix of Logistic Regression",
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