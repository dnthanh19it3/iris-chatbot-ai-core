# things we need for NLP
import os

from flask_cors import CORS
from nltk.stem.lancaster import LancasterStemmer
from underthesea import word_tokenize
from flask import Flask, request
from datetime import datetime

from ultils import getjson
import train

stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import random

# restore all of our data structures
import pickle
import json

class Predict:
    json_data = None
    model = None
    data = None
    classes = None
    words = None
    train_x = None
    train_y = None
    net = None
    context = {}
    ERROR_THRESHOLD = 0.25
    intents = None
    init_flag = 0
    def __init__(self):
        app = Flask(__name__)
        app.config["DEBUG"] = True
        CORS(app)
        this = self
        @app.route("/")
        def main():
            if (this.init_flag == 1):
                return json.dumps({"status": "1", "message": "Đã khởi chạy", "detail": "Kiểm tra máy chủ có đang hoạt động hay không"})
            else:
                return json.dumps({"status": "0", "message": "Chưa khởi chạy", "detail": "Kiểm tra máy chủ có đang hoạt động hay không"})
        @app.route("/<project_id>/predict")
        def predict(project_id):
            try:
                self.initPredictEngine(project_id)
            except Exception as e:
                print(e)
            input_data = request.args.get('msg')
            project_data = request.args.get('project_id')
            class_name = this.classify(input_data)
            response_content = this.response(input_data)

            command = 0

            if response_content == "" or response_content is None:
                command = 1

            output_data = {"type": command, "class": class_name, "response": response_content}
            return json.dumps(output_data)

        @app.route("/train_app/<project_id>")
        def train_app(project_id):
            try:
                train.train(project_id)
                return json.dumps({"status": "1"    , "message": "Train thành công!"})
            except Exception as e:
                return json.dumps({"status": "0", "message": "Train thất bại!", "detail": str(e)})
        app.run(host="0.0.0.0", port='5000')

    def initPredictEngine(self, project_id):
        try:
            startTime = datetime.now()
            self.data = pickle.load(open("training_data", "rb"))
            self.words = self.data['words']
            self.classes = self.data['classes']
            self.train_x = self.data['train_x']
            self.train_y = self.data['train_y']
            self.intents = getjson.getStaticJson(project_id)
            # Build neural network
            self.net = tflearn.input_data(shape=[None, len(self.train_x[0])])
            self.net = tflearn.fully_connected(self.net, 8)
            # self.net = tflearn.fully_connected(self.net, 8)
            self.net = tflearn.fully_connected(self.net, len(self.train_y[0]), activation='softmax')
            self.net = tflearn.regression(self.net)
            # Define model and setup tensorboard
            self.model = tflearn.DNN(self.net, tensorboard_dir='tflearn_logs')
            # load our saved model
            self.model.load('./model/' + project_id + '/model.tflearn')
            self.init_flag = 1
            endTime = datetime.now()
        except Exception as e:
            print(e)
            return 0
        return 1
    def clean_up_sentence(self, sentence):
        # tokenize the pattern
        sentence_words = word_tokenize(sentence)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words
    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(words)
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s:
                    bag[i] = 1

        return(np.array(bag))
# create a data structure to hold user context
    def classify(self, sentence):
        # generate probabilities from the model
        results = []
        try:
            results = self.model.predict([self.bow(sentence, self.words)])[0]
        except Exception as e:
            print(e)
            # self.initPredictEngine()
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], np.float64(r[1])))
        # return tuple of intent and probability
        return return_list

    def response(self, sentence, userID='123', show_details=False):
        results = self.classify(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            # if show_details: print('context:', i['context_set'])
                            self.context[userID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                            (userID in self.context and 'context_filter' in i and i['context_filter'] == self.context[userID]):
                            if(len(i['responses']) > 0):
                                return random.choice(i['responses'])
                            else:
                                return None
                results.pop(0)
        return results
app = Predict()



