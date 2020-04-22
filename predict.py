from joblib import load

# load the DictVectorizer and Random Forest Model
rf_model = joblib.load('student_rf_best_model.joblib')
dv_new = load('DictVectorizer.joblib')


def grabInput(media_group=0, primary_program=0, unsubscribed=0, application_fee_waived=0, age=29, attended_event=0):

    ls_keys = ['media_group', 'primary_program', 'unsubscribed', 'application_fee_waived', 'age', 'attended_event']

    ls_values = [media_group, primary_program, unsubscribed, application_fee_waived, age, attended_event]

    ziplist = zip(ls_keys, ls_values)

    output = dict(ziplist)

    return output


def predict():

    input = grabInput(media_group='Events', primary_program='Acting for Film', unsubscribed=0, application_fee_waived=0,
                      age=20, attended_event=1)

    x = dv_new.transform(input)

    y = rf_model.predict(x)

    y_proba = rf_model.predict_proba(x)

    if y[0] == 1:

        print('There is %.02f%% of the chance this lead is going to convert to opportunity' % (y_proba[0][1] * 100))

    elif y[0] == 0:

        print('There is %.02f%% of the chance this lead is not going to convert to opportunity' % (y_proba[0][1] * 100))


if __name__ == "__main__":
    predict()
