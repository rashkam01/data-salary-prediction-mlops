from predict import prepare_sal_features
from predict import predict

salary_features = {
    "Rating": 3.2,
    "Job Title": "Android Developer",
    "Location": "Bangalore",
    "Employment Status": "Full Time",
    "Job Roles": "Android",
}

features = prepare_sal_features(salary_features)
pred = predict(features)
print(pred)
