SYNTHETIC_FEATURE_COLUMN_PREFIX = "feature_"
GROUPING_COLUMN_NAME = "group"
RESPONSE_COLUMN_NAME = "response"

FIXED_EFFECT_PART_NAME = "[FIXED_EFFECT]"
RANDOM_EFFECT_PART_NAME = "[RANDOM_EFFECT]"
NOISE_PART_NAME = "[NOISE]"

INTERCEPT_COLUMN_NAME = "intercept"
ENCODED_COLUMN_PREFIX = "ENCODED_"
FIXED_ONLY_COLUMN_PREFIX = "FIXED_"
SPATIAL_COLUMN_PREFIX = "SPATIAL_COL_"
SPATIAL_ID_COLUMN_NAME = "SPATIAL_ID"

SPLIT_COLUMN_NAME = "split"
TRAIN_SPLIT_NAME = "TRAIN"
VALIDATION_SPLIT_NAME = "VALIDATION"
TEST_SPLIT_NAME = "TEST"

PREDICTION_SEED_OFFSETS = {
    VALIDATION_SPLIT_NAME: 101,
    TEST_SPLIT_NAME: 102,
}
EVALUATION_SEED_OFFSETS = {
    VALIDATION_SPLIT_NAME: 201,
    TEST_SPLIT_NAME: 202,
}

CONTEXT_ROLE_COLUMN_NAME = "context_role"
SUPPORT_ROLE_NAME = "SUPPORT"
TARGET_ROLE_NAME = "TARGET"

ALL_TASK_NAMES = ["in_context", "few_shot"]
IN_CONTEXT_TASK_NAME = "in_context"
FEW_SHOT_TASK_NAME = "few_shot"

# The hyperparameters that were tuned for each model type.
TREE_BOOSTING_TUNED_HYPERPARAMS = [
    "model.train_params.learning_rate",
    "model.train_params.lambda_l2",
    "model.train_params.min_data_in_leaf",
    "model.train_params.num_leaves",
    "model.train_params.max_bin",
    "model.train_params.max_depth",
]

NP_TUNED_HYPERPARAMS = [
    "model.params.dropout",
    "model.train_params.learning_rate",
]

NPBOOST_TUNED_HYPERPARAMS = [
    "model.np_params.dropout",
    "model.np_train_params.learning_rate",
    "model.np_train_params.epochs_per_round",
    "model.lgbm_params.learning_rate",
    "model.lgbm_params.lambda_l2",
    "model.lgbm_params.min_data_in_leaf",
    "model.lgbm_params.num_leaves",
    "model.lgbm_params.max_bin",
    "model.lgbm_params.max_depth",
]
# The best-model parameter CSVs contain additional metadata about top-performing models, 
# such as the optimal boosting round or epoch used for the restored checkpoint.
# These values are not model hyperparameters and must not be passed when re-running experiments.
TREE_BOOSTING_SELECTED_MODEL_PARAMS = [
    "best_boosting_round",
    *TREE_BOOSTING_TUNED_HYPERPARAMS,
]
NP_SELECTED_MODEL_PARAMS = [
    "epochs_trained",
    *NP_TUNED_HYPERPARAMS,
]
NPBOOST_SELECTED_MODEL_PARAMS = [
    "best_boosting_round",
    *NPBOOST_TUNED_HYPERPARAMS,
]

TUNED_HYPERPARAMS_PER_MODEL = {
    "lme": [],
    "gplinear": [],
    "tabicl_taskwise": [],
    "tabicl_pooled": [],
    "gbm_group_cat": TREE_BOOSTING_TUNED_HYPERPARAMS,
    "gbm_no_group": TREE_BOOSTING_TUNED_HYPERPARAMS,
    "np": NP_TUNED_HYPERPARAMS,
    "anp": NP_TUNED_HYPERPARAMS,
    "npboost": NPBOOST_TUNED_HYPERPARAMS,
    "anpboost": NPBOOST_TUNED_HYPERPARAMS,
}

SELECTED_MODEL_PARAMS_PER_MODEL = {
    "lme": [],
    "gplinear": [],
    "tabicl_taskwise": [],
    "tabicl_pooled": [],
    "gbm_group_cat": TREE_BOOSTING_SELECTED_MODEL_PARAMS,
    "gbm_no_group": TREE_BOOSTING_SELECTED_MODEL_PARAMS,
    "np": NP_SELECTED_MODEL_PARAMS,
    "anp": NP_SELECTED_MODEL_PARAMS,
    "npboost": NPBOOST_SELECTED_MODEL_PARAMS,
    "anpboost": NPBOOST_SELECTED_MODEL_PARAMS,
}
