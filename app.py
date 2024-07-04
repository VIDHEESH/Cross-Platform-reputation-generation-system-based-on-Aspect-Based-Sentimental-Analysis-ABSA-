from flask import Flask, render_template, request, jsonify
import sentiment_analysis as sa
# ...

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.json['sentence']
    sentiment = sa.predict_sentiment(sentence)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)