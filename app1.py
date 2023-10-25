
import numpy as np
from flask import Flask,render_template,request
import pickle
import keras
from keras.preprocessing import image

app=Flask(__name__)

model=pickle.load(open('models/model1.pkl','rb'))

@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('pagee1.html')

@app.route("/prediction", methods=['GET','POST'])
def prediction():
    img=request.files['filename']
    img.save("imgg.jpg")
    test_image=image.load_img(r"D:\Face Emotion Recognition\webpagetwo\imgg.jpg",target_size=(64,64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    if result[0][0]==1:
        prediction='Drenched in a touch of sadness.'
    else:
        prediction='Radiating happiness vibes!'
    return render_template('pagee1_bottom.html',prediction_text='You are '+prediction)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)