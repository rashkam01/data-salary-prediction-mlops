import requests

salary_features = {
    "Rating": 3.2,
    "Job Title": "Android Developer",
    "Location": "Bangalore",
    "Employment Status": "Full Time",
    "Job Roles": "Android",
}

url = "http://127.0.0.1:9696/predict_salary"
response = requests.post(url, json=salary_features)
print(response.json())
