################################################################################
# 0. 套件 ----------------------------------------------------------------------
################################################################################
pkgs <- c("tidyverse","data.table","keras","tensorflow","abind","kerastuneR","reticulate")
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install)
lapply(pkgs, library, character.only = TRUE)

################################################################################
# 1. 讀資料 --------------------------------------------------------------------
################################################################################
train <- fread("data/preparation/train.csv")
val   <- fread("data/preparation/validation.csv")
test  <- fread("data/preparation/test.csv")

id_col <- "stock_id"
seq_len <- 30
batch_size <- 128
max_epochs <- 40

feature_cols <- setdiff(names(train), c(id_col,"date","target"))
n_features   <- length(feature_cols)

################################################################################
# 2. 把 data.frame → 時序樣本 ---------------------------------------------------
################################################################################
make_seq <- function(dat){
  lst_X <- list(); lst_y <- list()
  for (d in split(dat, dat[[id_col]])) {
    d <- arrange(d, date)
    if (nrow(d) < seq_len) next
    idx_end <- seq_len(nrow(d))
    idx_end <- idx_end[idx_end >= seq_len]
    
    lst_X <- c(lst_X,
               lapply(idx_end, function(e)
                 as.matrix(d[(e-seq_len+1):e, ..feature_cols])))
    lst_y <- c(lst_y, d$target[idx_end])
  }
  X_raw <- abind::abind(lst_X, along = 3)        # 30 × F × n
  list(
    X = aperm(X_raw, c(3, 1, 2)),                # n × 30 × F
    y = matrix(as.numeric(unlist(lst_y)), ncol = 1)
  )
}


tr <- make_seq(train)
va <- make_seq(val)
te <- make_seq(test)

cat("Sequences  ➜ Train:", dim(tr$X)[1],
    "| Val:", dim(va$X)[1],
    "| Test:", dim(te$X)[1], "\n")

################################################################################
# 3. class weight (以 train 計) -------------------------------------------------
################################################################################
tbl <- table(tr$y)
class_wt <- list(
  "0" = (1 / tbl["0"]) * (length(tr$y) / 2),
  "1" = (1 / tbl["1"]) * (length(tr$y) / 2)
)

################################################################################
# 4. CNN + Bi-LSTM + Keras-tuner ----------------------------------------------
################################################################################
build_model <- function(hp){
  input <- layer_input(shape = c(seq_len, n_features))
  
  # ── Conv Block，可選 1~2 層 ───────────────────────────────
  conv_layers <- hp$Int("n_conv", 1, 2)
  x <- input
  for (j in seq_len(conv_layers)) {
    filters <- hp$Int(paste0("filters_", j), 32, 192, step = 32)
    x <- x %>% layer_conv_1d(filters = filters, kernel_size = 3,
                             activation = "relu", padding = "causal")
  }
  x <- x %>% layer_max_pooling_1d(pool_size = 2)
  
  # ── Bi-LSTM Block：1~3 層，每層 64-384 ───────────────────
  n_bi <- hp$Int("n_bi", 1, 3)
  for (i in seq_len(n_bi)) {
    units <- hp$Int(paste0("units_", i), 64, 384, step = 64)
    x <- x %>% bidirectional(layer_lstm(
      units          = units,
      return_sequences = (i < n_bi),
      kernel_regularizer = regularizer_l2(1e-4)))
  }
  
  x <- x %>% layer_dropout(rate = hp$Choice("drop", c(0.1,0.2,0.3,0.4,0.5))) %>% 
    layer_dense(units = hp$Int("fc", 32, 128, step = 16),
                activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid")
  
  model <- keras_model(input, x)
  model %>% compile(
    optimizer = optimizer_adam(
      learning_rate = hp$Choice("lr", c(1e-3,5e-4,1e-4,5e-5))
    ),
    loss   = "binary_crossentropy",
    metrics = "accuracy"
  )
  model
}


tuner <- RandomSearch(
  build_model,
  objective      = "val_accuracy",
  max_trials     = 100,          # ↑ 150 Trial
  executions_per_trial = 1,
  directory      = "tuner_dir",
  project_name   = "cnn_bilstm_stock"
)

tuner %>% fit_tuner(
  tr$X, tr$y,
  epochs          = max_epochs,
  batch_size      = batch_size,
  validation_data = list(va$X, va$y),
  callbacks       = list(callback_early_stopping(patience = 6, restore_best_weights = TRUE)),
  class_weight    = class_wt
)

best_hp_all <- tuner$get_best_hyperparameters()
best_hp     <- best_hp_all[[1]]

################################################################################
# 5. 用 Train+Val 重訓 ----------------------------------------------------------
################################################################################
X_tv <- abind(tr$X, va$X, along = 1)
y_tv <- rbind(tr$y, va$y)

final_mod <- build_model(best_hp)

final_mod %>% fit(
  X_tv, y_tv,
  epochs         = 60,
  batch_size     = batch_size,
  validation_split = 0,
  callbacks      = list(callback_early_stopping(patience = 10, restore_best_weights = TRUE)),
  class_weight   = class_wt
)

################################################################################
# 6. Test 評分 ------------------------------------------------------------------
################################################################################
score <- final_mod %>% evaluate(te$X, te$y, verbose=0)
cat("\n========================\n",
    "Test accuracy :", sprintf('%.4f', score["accuracy"]), "\n",
    "========================\n")
