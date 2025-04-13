from flask import Flask, render_template, request
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and tokenizer
model = load_model('final_caption_model.h5', compile=False)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 30  # set according to your model

# Load VGG16 for feature extraction
vgg = VGG16()
vgg = Model(vgg.input, vgg.layers[-2].output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return vgg.predict(x).reshape(1, 4096)

def generate_caption(img_path):
    feature = extract_features(img_path)
    caption = ['<start>']
    
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding='post')
        pred = model.predict([feature, seq], verbose=0)
        word_idx = np.argmax(pred)
        word = tokenizer.index_word.get(word_idx, '<unk>')
        if word == '<end>':
            break
        caption.append(word)
    
    return ' '.join(caption[1:])

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    if request.method == 'POST':
        f = request.files['image']
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        caption = generate_caption(filepath)
        return render_template('index.html', caption=caption, image_url=filepath)
    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
