import torch
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
import numpy as np
import traceback
app = Flask(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model_name = '/home/ubuntu/model'
model = SentenceTransformer(model_name).to(device)

targets = ['ask_question_ingredients_tools',
 'return_list_ingredients_tools',
 'request_next_step',
 'return_next_step',
 'ask_question_recipe_steps',
 'answer_question_recipe_steps',
 'answer_question_external_fact',
 'stop',
 'ask_student_question',
 'chitchat',
 'misc']
sentences2 = [' '.join(x.split('_')) for x in targets]

# Set the model to evaluation mode
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.json)
        

        # Generate model outputs
        with torch.no_grad():
            sentences1 = request.json['text']
            embedding1 = model.encode(sentences1)
            embedding2 = model.encode(sentences2)
            score = util.cos_sim(embedding1, embedding2).tolist()[0]
            predicted = targets[np.argmax(score)]

        # Extract the model output and return as JSON
        return jsonify({'statusCode': 200, 
                        "body": {
                                    "label": predicted,
                                    "score": score[np.argmax(score)]
                                }
        })
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'statusCode': 500, 
                        "body": tb
        })

@app.route('/', methods=['GET'])
def home():
    return jsonify({'statusCode': 200, 
                    "body": 'Use command: curl -X POST \
                    -H "Content-Type: application/json" \
                    -d \'{"text": "The first thing you should do is make sure you open the zipper as much as possible."}\' \
                    http://localhost:5001/predict'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
