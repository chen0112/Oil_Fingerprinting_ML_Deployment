from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def calculate():
    if request.method == "POST":
        biomarker = request.form.get("value00")
        if biomarker == "Terpanes":
            model1 = pickle.load(open('./artifacts/oil_type_prediction.pickle_Terpanes', 'rb'))
            feature1 = int(request.form.get("value01"))
            feature2 = int(request.form.get("value02"))
            feature3 = int(request.form.get('value03'))
            feature4 = int(request.form.get('value04'))
            feature5 = int(request.form.get('value05'))
            feature6 = int(request.form.get('value06'))
            feature7 = int(request.form.get('value07'))
            feature8 = int(request.form.get('value08'))
            feature9 = int(request.form.get('value09'))
            feature10 = int(request.form.get('value10'))
            feature11 = int(request.form.get('value11'))
            feature12 = int(request.form.get('value12'))
            feature13 = int(request.form.get('value13'))
            x = np.array([feature1, feature2, feature3,feature4,feature5,feature6,feature7, feature8, feature9, feature10, feature11, feature12, feature13])
            x = x.reshape(1,-1)
            output = model1.predict(x)
            return "result = " + str(output)
        elif biomarker == "Diamantanes":
            model2 = pickle.load(open('./artifacts/oil_type_prediction.pickle_Diamantanes', 'rb'))
            feature1 = int(request.form.get("value01"))
            feature2 = int(request.form.get("value02"))
            feature3 = int(request.form.get('value03'))
            feature4 = int(request.form.get('value04'))
            feature5 = int(request.form.get('value05'))
            x = np.array([feature1, feature2, feature3, feature4, feature5])
            x = x.reshape(1, -1)
            output = model2.predict(x)
            return "result = " + str(output)
        elif biomarker == "MA-steranes":
            model2= pickle.load(open('./artifacts/oil_type_prediction_by_MA_Steranes.pickle', 'rb'))
            feature1 = int(request.form.get("value01"))
            feature2 = int(request.form.get("value02"))
            feature3 = int(request.form.get('value03'))
            feature4 = int(request.form.get('value04'))
            feature5 = int(request.form.get('value05'))
            feature6 = int(request.form.get('value06'))
            feature7 = int(request.form.get('value07'))
            x = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7])
            x = x.reshape(1, -1)
            output = model2.predict(x)
            return "result = " + str(output)
        elif biomarker == "TA-steranes":
            model2= pickle.load(open('./artifacts/oil_type_prediction_by_TA_Steranes.pickle', 'rb'))
            feature1 = int(float(request.form.get("value01")))
            feature2 = int(float(request.form.get("value02")))
            feature3 = int(float(request.form.get("value03")))
            feature4 = int(float(request.form.get("value04")))
            feature5 = int(float(request.form.get("value05")))
            feature6 = int(float(request.form.get("value06")))
            feature7 = int(float(request.form.get("value07")))
            x = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7])
            x = x.reshape(1, -1)
            output = model2.predict(x)
            return "result = " + str(output)
        elif biomarker == "Steranes":
            model2= pickle.load(open('./artifacts/oil_type_prediction.pickle_Steranes', 'rb'))
            feature1 = int(request.form.get("value01"))
            feature2 = int(request.form.get("value02"))
            feature3 = int(request.form.get('value03'))
            feature4 = int(request.form.get('value04'))
            feature5 = int(request.form.get('value05'))
            feature6 = int(request.form.get('value06'))
            feature7 = int(request.form.get('value07'))
            feature8 = int(request.form.get('value08'))
            feature9 = int(request.form.get('value09'))
            feature10 = int(request.form.get('value10'))
            x = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7,feature8,feature9,feature10])
            x = x.reshape(1, -1)
            output = model2.predict(x)
            return "result = " + str(output)



    else:
        return "no correct request"


if __name__ == '__main__':
    app.run()
