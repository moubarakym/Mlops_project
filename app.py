from flask import Flask, request, jsonify,  render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Charger le modèle de machine learning
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])



@app.route("/", methods=["GET"])

def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        credit_lines_outstanding = int(request.form["credit_lines_outstanding"])
        loan_amt_outstanding = int(request.form["loan_amt_outstanding"])
        total_debt_outstanding = int(request.form["total_debt_outstanding"])
        income = float(request.form["income"])
        years_employed = int(request.form["years_employed"])
        fico_score = int(request.form["fico_score"])
        
        prediction = model.predict(
            [[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]]
        )

        if prediction[0] == 1:
            return render_template(
                "index.html",
                prediction_text="You are at risk of defaulting on your loan. Please contact your bank :(",
            )

        else:
            return render_template(
                "index.html", prediction_text="You are not at risk of defaulting on your loan :)"
            )

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
