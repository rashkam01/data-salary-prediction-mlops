import pickle
import xgboost as xgb
from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient


# with open("model.xgb", "rb") as f_in:
#     (dv, xgbmodel) = pickle.load(f_in)

# dv = pickle.load(open("backup/preprocessor.b", "rb"))
# loaded_model = xgb.Booster(model_file="backup/model.xgb")
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
RUN_ID = "7ff20c8c0dc24ed0b0cbca764d81bacd"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

path = client.download_artifacts(
    run_id=RUN_ID,
    path="/Users/rashmi/Documents/rashmi/github_repos/AtoZ_ml_ops_course/ds_salary_prediction/data-salary-prediction-mlops/02-workflow-orchestration/prefect-mlops-zoomcamp/mlruns/1/7ff20c8c0dc24ed0b0cbca764d81bacd/artifacts/preprocessor/preprocessor.b",
)

with open(path, "rb") as f_out:
    dv = pickle.load(f_out)


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("salary-predictor-exp")


logged_model = f"runs:/{RUN_ID}/models_mlflow"

loaded_model = mlflow.pyfunc.load_model(logged_model)
print(loaded_model)


def prepare_sal_features(salary_features):
    fea = {}
    fea["Rating"] = salary_features["Rating"]
    fea["Job Title"] = salary_features["Job Title"]
    fea["Location"] = salary_features["Location"]
    fea["Employment Status"] = salary_features["Employment Status"]
    fea["Job Roles"] = salary_features["Job Roles"]
    return fea


def predict_sal_new(all_features1):
    X_test = dv.transform(all_features1)
    # dmatrix = xgb.DMatrix(X_test)
    preds = loaded_model.predict(X_test)
    return float(preds[0])


app = Flask("salary-prediction")


@app.route("/predict_salary", methods=["POST"])
def predict_salary():
    try:
        job_features = request.get_json()
        if not job_features:
            return jsonify({"error": "Empty or invalid JSON data"})
        features = prepare_sal_features(job_features)
        preds = predict_sal_new(features)
        # X_test = dv.transform(features)
        # dmatrix = xgb.DMatrix(X_test)
        # preds = loaded_model.predict(dmatrix)
        # fl_preds = float(preds[0])
        # result = {"salary": fl_preds}
        result = {"salary": preds}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
