import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

flask_app = Flask(__name__)
model = joblib.load('logistic_regression.joblib')

#mean
mu_loan_amnt                          =8.920761
mu_loan_percent_income                =-1.954470
mu_person_income                      =10.872789
mu_loan_int_rate                      =2.354688
mu_person_age                         =26.723104
mu_person_emp_length                  =4.653295
mu_cb_person_cred_hist_length         =5.257881
mu_person_home_ownership_MORTGAGE     =0.397414
mu_person_home_ownership_OTHER        =0.003063
mu_person_home_ownership_OWN          =0.076786
mu_person_home_ownership_RENT         =0.522738
mu_loan_intent_DEBTCONSOLIDATION      =0.159357
mu_loan_intent_EDUCATION              =0.202663
mu_loan_intent_HOMEIMPROVEMENT        =0.109116
mu_loan_intent_MEDICAL                =0.187221
mu_loan_intent_PERSONAL               =0.166291
mu_loan_intent_VENTURE                =0.175352
mu_loan_grade_A                       =0.330370
mu_loan_grade_B                       =0.318799
mu_loan_grade_C                       =0.200834
mu_loan_grade_D                       =0.111967
mu_loan_grade_E                       =0.029225
mu_loan_grade_F                       =0.006806
mu_loan_grade_G                       =0.001999
mu_cb_person_default_on_file_N        =0.823797
mu_cb_person_default_on_file_Y        =0.176203

#standard deviation
sigma_loan_amnt                         =0.701507
sigma_loan_percent_income               =0.681557
sigma_person_income                     =0.489889
sigma_loan_int_rate                     =0.306691
sigma_person_age                        =4.499867
sigma_person_emp_length                 =3.824701
sigma_cb_person_cred_hist_length        =3.300048
sigma_person_home_ownership_MORTGAGE    =0.489363
sigma_person_home_ownership_OTHER       =0.055259
sigma_person_home_ownership_OWN         =0.266251
sigma_person_home_ownership_RENT        =0.499483
sigma_loan_intent_DEBTCONSOLIDATION     =0.366008
sigma_loan_intent_EDUCATION             =0.401983
sigma_loan_intent_HOMEIMPROVEMENT       =0.311785
sigma_loan_intent_MEDICAL               =0.390089
sigma_loan_intent_PERSONAL              =0.372342
sigma_loan_intent_VENTURE               =0.380268
sigma_loan_grade_A                      =0.470346
sigma_loan_grade_B                      =0.466011
sigma_loan_grade_C                      =0.400624
sigma_loan_grade_D                      =0.315325
sigma_loan_grade_E                      =0.168438
sigma_loan_grade_F                      =0.082220
sigma_loan_grade_G                      =0.044670
sigma_cb_person_default_on_file_N       =0.380993
sigma_cb_person_default_on_file_Y       =0.380993

#initialized
loan_amnt                         =0
loan_percent_income               =0
person_income                     =0
loan_int_rate                     =0
loan_status                       =0
person_age                        =0
person_emp_length                 =0
cb_person_cred_hist_length        =0
person_home_ownership_MORTGAGE    =0
person_home_ownership_OTHER       =0
person_home_ownership_OWN         =0
person_home_ownership_RENT        =0
loan_intent_DEBTCONSOLIDATION     =0
loan_intent_EDUCATION             =0
loan_intent_HOMEIMPROVEMENT       =0
loan_intent_MEDICAL               =0
loan_intent_PERSONAL              =0
loan_intent_VENTURE               =0
loan_grade_A                      =0
loan_grade_B                      =0
loan_grade_C                      =0
loan_grade_D                      =0
loan_grade_E                      =0
loan_grade_F                      =0
loan_grade_G                      =0
cb_person_default_on_file_N       =0
cb_person_default_on_file_Y       =0

def log_transformed(value):
    return np.log(value)

def standard_scaler(x, mu, sigma):
    return (x-mu)/sigma


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["GET", "POST"])
def predict():
    features = [x for x in request.form.values()]
    person_age = float(features[0])
    person_income = float(features[1])
    person_home_ownership = features[2]
    person_emp_length = float(features[3])
    loan_intent = features[4]
    loan_grade = features[5]
    loan_amnt = float(features[6])
    loan_int_rate = float(features[7])
    loan_percent_income = float(features[8])
    cb_person_default_on_file = features[9]
    cb_person_cred_hist_length = float(features[10])

    loan_amnt = log_transformed(loan_amnt)
    loan_percent_income = log_transformed(loan_percent_income)
    person_income = log_transformed(person_income)
    loan_int_rate = log_transformed(loan_int_rate)

    if person_home_ownership == "MORTGAGE":
        person_home_ownership_MORTGAGE = 1
        person_home_ownership_OTHER = 0
        person_home_ownership_OWN = 0
        person_home_ownership_RENT = 0
    elif person_home_ownership == "OTHER":
        person_home_ownership_MORTGAGE = 0
        person_home_ownership_OTHER = 1
        person_home_ownership_OWN = 0
        person_home_ownership_RENT = 0
    elif person_home_ownership == "OWN":
        person_home_ownership_MORTGAGE = 0
        person_home_ownership_OTHER = 0
        person_home_ownership_OWN = 1
        person_home_ownership_RENT = 0
    elif person_home_ownership == "RENT":
        person_home_ownership_MORTGAGE = 0
        person_home_ownership_OTHER = 0
        person_home_ownership_OWN = 0
        person_home_ownership_RENT = 1
    if loan_intent == "DEBTCONSOLIDATION":
        loan_intent_DEBTCONSOLIDATION = 1
        loan_intent_EDUCATION = 0
        loan_intent_HOMEIMPROVEMENT = 0
        loan_intent_MEDICAL = 0
        loan_intent_PERSONAL = 0
        loan_intent_VENTURE = 0
    elif loan_intent == "EDUCATION":
        loan_intent_DEBTCONSOLIDATION = 0
        loan_intent_EDUCATION = 1
        loan_intent_HOMEIMPROVEMENT = 0
        loan_intent_MEDICAL = 0
        loan_intent_PERSONAL = 0
        loan_intent_VENTURE = 0
    elif loan_intent == "HOMEIMPROVEMENT":
        loan_intent_DEBTCONSOLIDATION = 0
        loan_intent_EDUCATION = 0
        loan_intent_HOMEIMPROVEMENT = 1
        loan_intent_MEDICAL = 0
        loan_intent_PERSONAL = 0
        loan_intent_VENTURE = 0
    elif loan_intent == "MEDICAL":
        loan_intent_DEBTCONSOLIDATION = 0
        loan_intent_EDUCATION = 0
        loan_intent_HOMEIMPROVEMENT = 0
        loan_intent_MEDICAL = 1
        loan_intent_PERSONAL = 0
        loan_intent_VENTURE = 0
    elif loan_intent == "PERSONAL":
        loan_intent_DEBTCONSOLIDATION = 0
        loan_intent_EDUCATION = 0
        loan_intent_HOMEIMPROVEMENT = 0
        loan_intent_MEDICAL = 0
        loan_intent_PERSONAL = 1
        loan_intent_VENTURE = 0
    elif loan_intent == "VENTURE":
        loan_intent_DEBTCONSOLIDATION = 0
        loan_intent_EDUCATION = 0
        loan_intent_HOMEIMPROVEMENT = 0
        loan_intent_MEDICAL = 0
        loan_intent_PERSONAL = 0
        loan_intent_VENTURE = 1
    if loan_grade == "A":
        loan_grade_A = 1
        loan_grade_B = 0
        loan_grade_C = 0
        loan_grade_D = 0
    elif loan_grade == "B":
        loan_grade_A = 0
        loan_grade_B = 1
        loan_grade_C = 0
        loan_grade_D = 0
    elif loan_grade == "C":
        loan_grade_A = 0
        loan_grade_B = 0
        loan_grade_C = 1
        loan_grade_D = 0
    elif loan_grade == "D":
        loan_grade_A = 0
        loan_grade_B = 0
        loan_grade_C = 0
        loan_grade_D = 1
    if cb_person_default_on_file == "N":
        cb_person_default_on_file_N = 1
        cb_person_default_on_file_Y = 0
    elif cb_person_default_on_file == "Y":
        cb_person_default_on_file_N = 0
        cb_person_default_on_file_Y = 1

    z_loan_amnt = standard_scaler(x=loan_amnt, mu=mu_loan_amnt, sigma=sigma_loan_amnt)
    z_loan_percent_income = standard_scaler(x=loan_percent_income, mu=mu_loan_percent_income,
                                            sigma=sigma_loan_percent_income)
    z_person_income = standard_scaler(x=person_income, mu=mu_person_income, sigma=sigma_person_income)
    z_loan_int_rate = standard_scaler(x=loan_int_rate, mu=mu_loan_int_rate, sigma=sigma_loan_int_rate)
    z_person_age = standard_scaler(x=person_age, mu=mu_person_age, sigma=sigma_person_age)
    z_person_emp_length = standard_scaler(x=person_emp_length, mu=mu_person_emp_length, sigma=sigma_person_emp_length)
    z_cb_person_cred_hist_length = standard_scaler(x=cb_person_cred_hist_length, mu=mu_cb_person_cred_hist_length,
                                                   sigma=sigma_cb_person_cred_hist_length)
    z_person_home_ownership_MORTGAGE = standard_scaler(x=person_home_ownership_MORTGAGE,
                                                       mu=mu_person_home_ownership_MORTGAGE,
                                                       sigma=sigma_person_home_ownership_MORTGAGE)
    z_person_home_ownership_OTHER = standard_scaler(x=person_home_ownership_OTHER, mu=mu_person_home_ownership_OTHER,
                                                    sigma=sigma_person_home_ownership_OTHER)
    z_person_home_ownership_OWN = standard_scaler(x=person_home_ownership_OWN, mu=mu_person_home_ownership_OWN,
                                                  sigma=sigma_person_home_ownership_OWN)
    z_person_home_ownership_RENT = standard_scaler(x=person_home_ownership_RENT, mu=mu_person_home_ownership_RENT,
                                                   sigma=sigma_person_home_ownership_RENT)
    z_loan_intent_DEBTCONSOLIDATION = standard_scaler(x=loan_intent_DEBTCONSOLIDATION,
                                                      mu=mu_loan_intent_DEBTCONSOLIDATION,
                                                      sigma=sigma_loan_intent_DEBTCONSOLIDATION)
    z_loan_intent_EDUCATION = standard_scaler(x=loan_intent_EDUCATION, mu=mu_loan_intent_EDUCATION,
                                              sigma=sigma_loan_intent_EDUCATION)
    z_loan_intent_HOMEIMPROVEMENT = standard_scaler(x=loan_intent_HOMEIMPROVEMENT, mu=mu_loan_intent_HOMEIMPROVEMENT,
                                                    sigma=sigma_loan_intent_HOMEIMPROVEMENT)
    z_loan_intent_MEDICAL = standard_scaler(x=loan_intent_MEDICAL, mu=mu_loan_intent_MEDICAL,
                                            sigma=sigma_loan_intent_MEDICAL)
    z_loan_intent_PERSONAL = standard_scaler(x=loan_intent_PERSONAL, mu=mu_loan_intent_PERSONAL,
                                             sigma=sigma_loan_intent_VENTURE)
    z_loan_intent_VENTURE = standard_scaler(x=loan_intent_VENTURE, mu=mu_loan_intent_VENTURE,
                                            sigma=sigma_loan_intent_VENTURE)
    z_loan_grade_A = standard_scaler(x=loan_grade_A, mu=mu_loan_grade_A, sigma=sigma_loan_grade_A)
    z_loan_grade_B = standard_scaler(x=loan_grade_B, mu=mu_loan_grade_B, sigma=sigma_loan_grade_B)
    z_loan_grade_C = standard_scaler(x=loan_grade_C, mu=mu_loan_grade_C, sigma=sigma_loan_grade_C)
    z_loan_grade_D = standard_scaler(x=loan_grade_D, mu=mu_loan_grade_D, sigma=sigma_loan_grade_D)
    z_loan_grade_E = standard_scaler(x=loan_grade_E, mu=mu_loan_grade_E, sigma=sigma_loan_grade_E)
    z_loan_grade_F = standard_scaler(x=loan_grade_F, mu=mu_loan_grade_F, sigma=sigma_loan_grade_F)
    z_loan_grade_G = standard_scaler(x=loan_grade_G, mu=mu_loan_grade_G, sigma=sigma_loan_grade_G)
    z_cb_person_default_on_file_N = standard_scaler(x=cb_person_default_on_file_N, mu=mu_cb_person_default_on_file_N,
                                                    sigma=sigma_cb_person_default_on_file_N)
    z_cb_person_default_on_file_Y = standard_scaler(x=cb_person_default_on_file_Y, mu=mu_cb_person_default_on_file_Y,
                                                    sigma=sigma_cb_person_default_on_file_Y)

    standard_scaled = [z_loan_amnt, z_loan_percent_income, z_person_income, z_loan_int_rate, z_person_age,
                       z_person_emp_length, z_cb_person_cred_hist_length, z_person_home_ownership_MORTGAGE,
                       z_person_home_ownership_OTHER, z_person_home_ownership_OWN, z_person_home_ownership_RENT,
                       z_loan_intent_DEBTCONSOLIDATION, z_loan_intent_EDUCATION, z_loan_intent_HOMEIMPROVEMENT,
                       z_loan_intent_MEDICAL, z_loan_intent_PERSONAL, z_loan_intent_VENTURE, z_loan_grade_A,
                       z_loan_grade_B, z_loan_grade_C,
                       z_loan_grade_D, z_loan_grade_E, z_loan_grade_F, z_loan_grade_G, z_cb_person_default_on_file_N,
                       z_cb_person_default_on_file_Y]

    prediction = model.predict([standard_scaled])
    if prediction == 0:
        prediction = "Non-Defaulter"
    elif prediction == 1:
        prediction = "Defaulter"
    return render_template("index.html", prediction_text = "{}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)