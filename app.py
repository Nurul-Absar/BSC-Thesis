from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizer import custom_tokenizer

app = Flask(__name__)

# Load the SVM model and vectorizer
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None

    if request.method == 'POST':
        input_text = request.form['input_text']

        # Preprocess input text using the same tokenizer as during training
        preprocessed_text = ' '.join(custom_tokenizer(input_text))

        # Transform preprocessed text using the loaded vectorizer
        input_vector = vectorizer.transform([preprocessed_text])

        # Make a prediction using the loaded SVM model
        prediction = svm_model.predict(input_vector)
        #class_labels = ['not bully', 'religious', 'threat', 'troll', 'sexual']
        predicted_class = prediction


    return render_template('index.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
