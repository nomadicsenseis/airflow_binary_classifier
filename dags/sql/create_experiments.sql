CREATE TABLE IF NOT EXISTS experiments (
    experiment_id SERIAL PRIMARY KEY,
    experiment_datetime VARCHAR NOT NULL,
    best_auc NUMERIC NOT NULL,
    best_xgb_n_estimators INTEGER NOT NULL,
    best_xgb_max_depth INTEGER NOT NULL,
    best_xgb_min_child_weight INTEGER NOT NULL,
    best_xgb_gamma NUMERIC NOT NULL,
    best_xgb_learning_rate NUMERIC NOT NULL,
    best_xgb_subsample NUMERIC NOT NULL,
    best_xgb_colsample_bytree NUMERIC NOT NULL,
    best_xgb_reg_alpha NUMERIC NOT NULL,
    best_xgb_reg_lambda NUMERIC NOT NULL
);