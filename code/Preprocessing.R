################################################################################
# 0. 套件 -----------------------------------------------------------------------
################################################################################
packages <- c("tidyverse", "data.table", "TTR", "lubridate", "scales")
to_install <- setdiff(packages, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install)
lapply(packages, library, character.only = TRUE)

################################################################################
# 1. 參數 & 資料夾 --------------------------------------------------------------
################################################################################
train_ratio <- 0.8
val_ratio   <- 0.1          # test = 1 - (train+val)

base_path   <- "data/tw_stock"
stock_info  <- fread("./台股科技優質股清單.csv")   # 需有 Stock_ID, Company_Name
available   <- list.dirs(base_path, recursive = FALSE, full.names = FALSE)

stock_info  <- stock_info %>% filter(as.character(Stock_ID) %in% available)
stock_paths <- deframe(stock_info %>% 
                         mutate(folder_path = file.path(base_path, Stock_ID)) %>% 
                         transmute(Stock_ID = as.character(Stock_ID), folder_path))

################################################################################
# 2. 公用函式 -------------------------------------------------------------------
################################################################################
tw2date <- function(mdate){
  x <- str_split(as.character(mdate), "/", simplify = TRUE)
  lubridate::ymd(sprintf("%d-%s-%s", as.integer(x[,1])+1911, x[,2], x[,3]))
}

to_numeric <- function(v) as.numeric(gsub(",", "", v))

clean_one_stock <- function(path, sid){
  # 1) 合併日 CSV
  df <- rbindlist(lapply(list.files(path, "\\.csv$", full.names = TRUE), fread), fill = TRUE)
  names(df) <- trimws(names(df))
  df <- as_tibble(df) %>% 
    rename(date = 日期, volume = 成交股數, value = 成交金額,
           open = 開盤價, high = 最高價, low = 最低價, close = 收盤價,
           change = 漲跌價差, trades = 成交筆數) %>% 
    mutate(date   = tw2date(date),
           volume = to_numeric(volume),
           value  = to_numeric(value),
           trades = to_numeric(trades),
           change = as.numeric(gsub("[^0-9.-]", "", change)),
           across(c(open, high, low, close), to_numeric)) %>% 
    arrange(date) %>% 
    filter(!is.na(close))
  
  # 2) 技術指標
  df <- df %>% 
    mutate(rsi14 = RSI(close, 14),
           ma5   = SMA(close, 5),
           ma20  = SMA(close, 20))          # 加一組較長 MA
  macd <- MACD(df$close, 12, 26, 9)
  df$macd_diff <- if (is.null(macd)) NA_real_ else macd[, "macd"] - macd[, "signal"]
  
  # 3) 衍生特徵 + outlier 裁剪
  df <- df %>% 
    mutate(volume_change = (volume - lag(volume)) / lag(volume),
           return_pct    = (close  - lag(close)) / lag(close),
           gap_pct       = (open   - lag(close)) / lag(close)) %>% 
    mutate(across(c(volume_change, return_pct, gap_pct),
                  ~ scales::squish(.x, quantile(.x, c(.01, .99), na.rm=TRUE)))) 
  
  # 4) target
  df <- df %>% mutate(target = as.integer(lead(close) > close)) %>% drop_na()
  
  df$stock_id <- sid
  df
}

################################################################################
# 3. 逐檔股票前處理 -------------------------------------------------------------
################################################################################
all_list <- lapply(names(stock_paths), function(sid){
  clean_one_stock(stock_paths[[sid]], sid)
})

# ── 放在 3. 前或 4. 前皆可 ──────────────────────────────────────────────────
add_tsmc_feature <- function(df_list, leader_id = "2330"){
  # 1) 取出台積電資料並算昨收
  tsmc <- dplyr::bind_rows(df_list) %>% 
    filter(stock_id == leader_id) %>% 
    arrange(date) %>% 
    mutate(tsmc_prev_close = lag(close)) %>% 
    select(date, tsmc_open = open, tsmc_prev_close)
  
  # 2) 左接回每檔 df，並算 (今開 – 昨收) / 昨收
  lapply(df_list, function(d){
    left_join(d, tsmc, by = "date") %>% 
      mutate(tsmc_gap_pct = (tsmc_open - tsmc_prev_close) / tsmc_prev_close)
  })
}
all_list <- add_tsmc_feature(all_list, leader_id = "2330")
################################################################################
# 4. 每檔先切 Train/Val/Test，再標準化 -----------------------------------------
################################################################################
split_and_scale <- function(df_one){
  n <- nrow(df_one)
  tr_end <- floor(n * train_ratio)
  va_end <- floor(n * (train_ratio + val_ratio))
  
  # 分割
  df_one <- df_one %>% arrange(date)
  tr <- df_one[1:tr_end, ]
  va <- df_one[(tr_end+1):va_end, ]
  te <- df_one[(va_end+1):n, ]
  
  # 找數值欄（扣掉 id/date/target）
  num_cols <- setdiff(names(df_one), c("stock_id","date","target"))
  
  mu <- colMeans(tr[, num_cols], na.rm = TRUE)
  sd <- sapply(tr[, num_cols], sd, na.rm = TRUE)
  sd[sd == 0] <- 1
  
  scale_df <- function(d){
    d[, num_cols] <- sweep(d[, num_cols], 2, mu, "-")
    d[, num_cols] <- sweep(d[, num_cols], 2, sd, "/")
    d
  }
  
  list(train = scale_df(tr),
       val   = scale_df(va),
       test  = scale_df(te))
}

splits <- lapply(all_list, split_and_scale)

train_df <- bind_rows(lapply(splits, `[[`, "train"))
val_df   <- bind_rows(lapply(splits, `[[`, "val"))
test_df  <- bind_rows(lapply(splits, `[[`, "test"))

################################################################################
# 5. 輸出 ----------------------------------------------------------------------
################################################################################
dir.create("data/preparation", showWarnings = FALSE, recursive = TRUE)
fwrite(train_df, "data/preparation/train.csv")
fwrite(val_df,   "data/preparation/validation.csv")
fwrite(test_df,  "data/preparation/test.csv")

cat("輸出完成：",
    nrow(train_df), "train |",
    nrow(val_df),   "val |",
    nrow(test_df),  "test\n")
