from sklearn import metrics
from teradataml import DataFrame, copy_to_sql
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib
import json
import numpy as np
import pandas as pd


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    model = joblib.load(f"{context.artifact_input_path}/model.joblib")

    feature_names = context.dataset_info.feature_names

    print("feature names: ", feature_names)
    
    target_name = context.dataset_info.target_names[0]

    print("target_name: ", target_name)

    print("dataset info sql: ", context.dataset_info.sql)

    test_df = DataFrame.from_query(context.dataset_info.sql)

    print("test data in td dataframe: ", test_df)

    test_pdf = test_df.to_pandas(all_rows=True)

    print("test data in pandas: ", test_pdf)

    X_test = test_pdf[feature_names]
    y_test = test_pdf[target_name]

    print("The X_test: ", X_test)
    print("The y_test: ", y_test)

    #errorbalanceOrig, errorBalanceDest, amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest

    X_test = X_test.astype({"step": 'int8', "CASH_OUT": 'int8', "TRANSFER": 'int8', "errorbalanceOrig": 'float32', "errorBalanceDest": 'float32', "amount": 'float32', "oldbalanceOrig": 'float32', "newbalanceOrig": 'float32', "oldbalanceDest": 'float32', "newbalanceDest": 'float32'})

    print("shape of the X_test: ", X_test.shape)
    
    y_test = y_test.astype({"isFraud": 'int8'})

    print("shape of the y_test: ", y_test.shape)

    print("Scoring the txns...")
    y_pred = model.predict(X_test)

    print("Shape of the y_pred: ", y_pred.shape)

    print("y_prd values: ", y_pred)

    y_pred_tdf = pd.DataFrame(y_pred, columns=[target_name])
    y_pred_tdf["txn_id"] = test_pdf["txn_id"].values

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    metrics.plot_confusion_matrix(model, X_test, y_test)
    save_plot('Confusion Matrix', context=context)

    metrics.plot_roc_curve(model, X_test, y_test)
    save_plot('ROC Curve', context=context)

    from xgboost import plot_importance
    model["xgb"].get_booster().feature_names = feature_names
    plot_importance(model["xgb"].get_booster(), max_num_features=10)
    save_plot("feature_importance.png", context=context)

    feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")

    predictions_table = "predictions_tmp"
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace")
    
    record_evaluation_stats(features_df=test_df,
                            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
                            feature_importance=feature_importance,
                            context=context)

    print("All done with txns evaluations!")
