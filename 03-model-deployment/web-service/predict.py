import pickle
import xgboost as xgb

# from flask import Flask, request, jsonify

# with open("model.xgb", "rb") as f_in:
#     (dv, xgbmodel) = pickle.load(f_in)

dv = pickle.load(open("preprocessor.b", "rb"))
model = xgb.Booster(model_file="model.xgb")


def prepare_sal_features(salary_features):
    fea = {}
    fea["Rating"] = salary_features["Rating"]
    fea["Job Title"] = salary_features["Job Title"]
    fea["Location"] = salary_features["Location"]
    fea["Employment Status"] = salary_features["Employment Status"]
    fea["Job Roles"] = salary_features["Job Roles"]
    return fea


def predict(features1):
    X_test = dv.transform(features1)
    dmatrix = xgb.DMatrix(X_test)
    preds = model.predict(dmatrix)
    return preds[0]
