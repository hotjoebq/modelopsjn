from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
from teradataml import DataFrame
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib

def train(context: ModelContext, **kwargs):

    print("function train starting...")
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    print("Feature Names: ", feature_names)
    print("Target Names: ", target_name)
    
    print("data set info: ", context.dataset_info.sql)

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    print(train_pdf)

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    print("Starting training the txns...")

    # fit model to training data
    # no scalers here
    model = Pipeline(steps=[('xgb', XGBClassifier(eta=context.hyperparams["eta"], max_depth=context.hyperparams["max_depth"], enable_categorical=True))])

    model.fit(X_train, y_train)

    print("Finished training txns...")

    # export model artifacts
    joblib.dump(model, f"{context.artifact_output_path}/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name,
                    pmml_f_name=f"{context.artifact_output_path}/model.pmml")
    

    print("Saved trained txns model")

    from xgboost import plot_importance
    model["xgb"].get_booster().feature_names = feature_names
    plot_importance(model["xgb"].get_booster(), max_num_features=10)
    save_plot("feature_importance.png", context=context)

    feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")

    print("Recording txns training stats......")
    
    categories = ["isFraud", "CASH_OUT", "TRANSFER"]

    record_training_stats(train_df,
                          features=feature_names,
                          targets=[target_name],
                          categorical=categories,
                          feature_importance=feature_importance,
                          context=context)
    
    print("All done with training txns!")
