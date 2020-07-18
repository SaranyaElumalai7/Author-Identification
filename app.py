from flask import render_template, Flask, request,url_for
from keras.models import load_model
import pickle 
import tensorflow as tf
graph = tf.get_default_graph()
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla = load_model('author.h5')
cla.compile(optimizer='adam',loss='categorical_crossentropy')
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/tpredict')
@app.route('/', methods = ['GET','POST'])
def page2():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        topic = request.form['text']
        print("Hey " +topic)
        topic=cv.transform([topic])
        print("\n"+str(topic.shape)+"\n")
        with graph.as_default():
            y_pred = cla.predict_classes(topic)
            print("pred is "+str(y_pred))
        if(y_pred==[0]):
            y_p="EAP"
        elif(y_pred==[1]):
            y_p="HPL"
        else:
            y_p="MWS"
        

        return render_template('index.html',ypred = y_p)
        



if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
    
