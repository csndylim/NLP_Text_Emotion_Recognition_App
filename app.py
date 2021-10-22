from flask import Flask, request
from flask.templating import render_template

import torch

import regex
from string import punctuation

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
# nltk.download('stopwords')

from transformers import RobertaModel, RobertaTokenizer


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("model/roberta_sent.bin")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 13)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def get_inputs(text, tokenizer):

    inps = tokenizer.encode_plus(
                list(text),
                None,
                add_special_tokens=True,
                max_length=256,
                pad_to_max_length=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
    
    inp_tok = inps['input_ids'].to('cpu', dtype = torch.long)
    ids = inps['attention_mask'].to('cpu', dtype = torch.long)
    segments = inps['token_type_ids'].to('cpu', dtype = torch.long)

    return inp_tok, ids, segments


def cleaner(user_text):

  wordnet_lemmatizer = WordNetLemmatizer()

  stop = stopwords.words('english')

  for punct in punctuation:
      stop.append(punct)

  def clean(x, stop_words):
      word_tokens = WordPunctTokenizer().tokenize(x.lower())
      x = [regex.sub(u'\p{^Latin}', u'', w) for w in word_tokens if w.isalpha() and len(w) > 2]
      return " ".join(x)

  cleaned_user = clean(user_text, stop) 
  
  return cleaned_user


def get_sentiment(user_input):

  # Define empty list to store emotions
  sentiment = []

  user_input = cleaner(user_input)

  # Pass user input to encode text
  inp_tok, ids, segments = get_inputs(user_input, tokenizer2)

  # Get predictions
  preds = saved_model2(inp_tok, ids, segments)

  # Get max pred
  # big_val, big_idx = torch.max(preds.data, dim=1)
  big_val, big_idx = torch.topk(preds.data, k=3, dim=1)

  for x in range(len(big_idx[0])):
    sentiment.append(list(labels_dict.keys())[list(labels_dict.values()).index(big_idx[0][x].item())])

  return sentiment


saved_model2 = torch.load('model/roberta_sent.bin', map_location=torch.device('cpu'))
tokenizer2 = RobertaTokenizer.from_pretrained("model/")

labels_dict = {'anger': 12,
 'boredom': 10,
 'empty': 0,
 'enthusiasm': 2,
 'fun': 7,
 'happiness': 9,
 'hate': 8,
 'love': 6,
 'neutral': 3,
 'relief': 11,
 'sadness': 1,
 'surprise': 5,
 'worry': 4}

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('from_ex.html')

@app.route('/', methods=['POST'])
def my_form_post():
    user_input = request.form['u']
    # user_input = 'missed the bus.... shit!'
    sentiment = get_sentiment(user_input)
    #print(sentiment)
    
    return str(sentiment)

# @app.route('/')
# def hello():


#     return "<h1>Hello World!</h1>" \
#            "\nThis is my introduction to Flask!" \
#            "\nI can write a lot of things on this page.\nLet's get started!"

if __name__ == '__main__':
    app.run()