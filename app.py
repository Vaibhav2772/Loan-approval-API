from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('loan_model.pkl')

GENDER_MAP = {'Female': 0, 'Male': 1}
MARRIED_MAP = {'No': 0, 'Yes': 1}
EDUCATION_MAP = {'Graduate': 0, 'Not Graduate': 1}
SELF_EMPLOYED_MAP = {'No': 0, 'Yes': 1}
PROPERTY_AREA_MAP = {'Metro': 0, 'Rural': 1, 'Semiurban': 2, 'Urban': 3, 'semi-urban': 4}


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "message": "Loan Approval Prediction API",
        "endpoint": "POST /predict"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        gender = GENDER_MAP.get(data['Gender'])
        married = MARRIED_MAP.get(data['Married'])
        education = EDUCATION_MAP.get(data['Education'])
        self_employed = SELF_EMPLOYED_MAP.get(data['Self_Employed'])
        property_area = PROPERTY_AREA_MAP.get(data['Property_Area'])

        if None in [gender, married, education, self_employed, property_area]:
            return jsonify({
                "error": "Invalid category value provided.",
                "valid_options": {
                    "Gender": list(GENDER_MAP.keys()),
                    "Married": list(MARRIED_MAP.keys()),
                    "Education": list(EDUCATION_MAP.keys()),
                    "Self_Employed": list(SELF_EMPLOYED_MAP.keys()),
                    "Property_Area": list(PROPERTY_AREA_MAP.keys())
                }
            }), 400

        features = pd.DataFrame([{
            'Gender': gender,
            'Married': married,
            'Dependents': int(data['Dependents']),
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': float(data['ApplicantIncome']),
            'CoapplicantIncome': float(data['CoapplicantIncome']),
            'LoanAmount': float(data['LoanAmount']),
            'Loan_Amount_Term': float(data['Loan_Amount_Term']),
            'Credit_History': float(data['Credit_History']),
            'Property_Area': property_area
        }])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        result = "Approved" if prediction == 1 else "Not Approved"
        confidence = round(float(max(probability)) * 100, 2)

        return jsonify({
            "prediction": result,
            "confidence_percent": confidence,
            "raw_prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
