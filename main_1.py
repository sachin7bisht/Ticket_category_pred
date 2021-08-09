import numpy as np
from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import math
import preprocess
import vect_file

app = Flask(__name__)
model = pickle.load(open('rf_w2v_.pkl', 'rb'))
num_model = pickle.load(open('minmax_scale_cs2_new_07.pkl', 'rb'))
# txt_model = pickle.load(open('tfidf.pkl', 'rb'))

# from gensim.models import Word2Vec

@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':

        category = request.form["category"]
        if category == 'category_0':
            category_0 = 1
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_1':
            category_0 = 0
            category_1 = 1
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_2':
            category_0 = 0
            category_1 = 0
            category_2 = 1
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_3':
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 1
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_4':
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 1
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_5':
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 1
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_6':
            category_0 = 0
            category_1 = 1
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 1
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_7':
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 1
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_8':
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 1
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_9':
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 1
            category_10 = 0
            category_11 = 0
            category_12 = 0

        elif category == 'category_10':
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 1
            category_11 = 0
            category_12 = 0

        elif category == 'category_11':
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 1
            category_12 = 0

        elif category == 'category_12':
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 1

        else:
            category_0 = 0
            category_1 = 0
            category_2 = 0
            category_3 = 0
            category_4 = 0
            category_5 = 0
            category_6 = 0
            category_7 = 0
            category_8 = 0
            category_9 = 0
            category_10 = 0
            category_11 = 0
            category_12 = 0

        sub_category = request.form["sub_category"]
        if sub_category == 'sub_category_1':
            sub_category_1 = 1
            sub_category_2 = 0
            sub_category_3 = 0
            sub_category_4 = 0
            sub_category_5 = 0
            sub_category_6 = 0

        elif category == 'sub_category_2':
            sub_category_1 = 0
            sub_category_2 = 1
            sub_category_3 = 0
            sub_category_4 = 0
            sub_category_5 = 0
            sub_category_6 = 0

        elif category == 'sub_category_3':
            sub_category_1 = 0
            sub_category_2 = 0
            sub_category_3 = 1
            sub_category_4 = 0
            sub_category_5 = 0
            sub_category_6 = 0

        elif category == 'sub_category_4':
            sub_category_1 = 0
            sub_category_2 = 0
            sub_category_3 = 0
            sub_category_4 = 1
            sub_category_5 = 0
            sub_category_6 = 0

        elif category == 'sub_category_5':
            sub_category_1 = 0
            sub_category_2 = 0
            sub_category_3 = 0
            sub_category_4 = 0
            sub_category_5 = 1
            sub_category_6 = 0

        elif category == 'sub_category_6':
            sub_category_1 = 0
            sub_category_2 = 0
            sub_category_3 = 0
            sub_category_4 = 0
            sub_category_5 = 0
            sub_category_6 = 1

        else:
            sub_category_1 = 0
            sub_category_2 = 0
            sub_category_3 = 0
            sub_category_4 = 0
            sub_category_5 = 0
            sub_category_6 = 0

        Impact = request.form["Impact"]
        if Impact == 'Impact_0':
            impact_0 = 1
            impact_1 = 0
            impact_2 = 0
            impact_3 = 0
            impact_4 = 0

        elif Impact == 'Impact_1':
            impact_0 = 0
            impact_1 = 1
            impact_2 = 0
            impact_3 = 0
            impact_4 = 0

        elif Impact == 'Impact_2':
            impact_0 = 0
            impact_1 = 0
            impact_2 = 1
            impact_3 = 0
            impact_4 = 0

        elif Impact == 'Impact_3':
            impact_0 = 0
            impact_1 = 0
            impact_2 = 0
            impact_3 = 1
            impact_4 = 0

        elif Impact == 'Impact_4':
            impact_0 = 0
            impact_1 = 0
            impact_2 = 0
            impact_3 = 0
            impact_4 = 1

        else:
            impact_0 = 0
            impact_1 = 0
            impact_2 = 0
            impact_3 = 0
            impact_4 = 0

        Urgency = request.form["Urgency"]
        if Urgency == 'Urgency_0':
            urgency_0 = 1
            urgency_1 = 0
            urgency_2 = 0
            urgency_3 = 0

        elif Urgency == 'Urgency_1':
            urgency_0 = 0
            urgency_1 = 1
            urgency_2 = 0
            urgency_3 = 0

        elif Urgency == 'Urgency_2':
            urgency_0 = 0
            urgency_1 = 0
            urgency_2 = 1
            urgency_3 = 0

        elif Urgency == 'Urgency_3':
            urgency_0 = 0
            urgency_1 = 0
            urgency_2 = 0
            urgency_3 = 1

        else:
            urgency_0 = 0
            urgency_1 = 0
            urgency_2 = 0
            urgency_3 = 0

        body = request.form['body']
        title = request.form['title']

        title_count = len([w for w in title.split() if w.isalnum() == True])
        body_count = len([w for w in body.split() if w.isalnum() == True])

        all = ' '.join([title, body])
        txt = preprocess.preprocess(all)
        length = len([w for w in txt.split() if w.isalnum() == True])

        num_attrib = [title_count, body_count, length]
        num_attrib = num_model.transform([num_attrib])
        title_count = num_attrib[0][0]
        body_count = num_attrib[0][1]
        length = num_attrib[0][2]

        txt_df = vect_file.w2v(txt)
        # df_1 = pd.DataFrame(txt_df)
        # df_1 = np.asarray(df_1).ravel()

        df_2 = np.asarray([
            title_count, body_count, length, category_0, category_1, category_2, category_3, category_4, category_5,
            category_6, category_7, category_8, category_9, category_10, category_11, category_12, sub_category_1, sub_category_2,
            sub_category_3, sub_category_4, sub_category_5, sub_category_6, urgency_0, urgency_1, urgency_2, urgency_3,
            impact_0, impact_1, impact_2, impact_3, impact_4])

        final_df = np.concatenate((txt_df, df_2), axis=0)
        ans=model.predict([final_df])
        output = ans[0]

        # predict_final = model_meta.predict(total_pred)
        # # output=predict_final
        # output = math.floor(abs(predict_final))

        return render_template('home.html', Prediction="Ticket Category is {}".format(output))

    return render_template("home.html")


if __name__ == "__main__":
    app.run()
