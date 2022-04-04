############################################################
# Import libraries
############################################################
import os
from flask import Flask, request, flash
from flask.templating import render_template

import torch

import regex
from string import punctuation
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
# nltk.download('stopwords')

from transformers import RobertaModel, RobertaTokenizer, MarianMTModel, MarianTokenizer, MarianConfig
import sentencepiece
from textblob import TextBlob
############################################################
# Import models
############################################################
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

############################################################
# Read the text input and clean before model testing
############################################################
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

  # Cleanse the data
  user_input = cleaner(user_input)

  # Pass user input to encode text
  inp_tok, ids, segments = get_inputs(user_input, tokenizer_emotion)

  # Get predictions
  preds = model_emotion(inp_tok, ids, segments)

  # Get max pred
  # big_val, big_idx = torch.max(preds.data, dim=1)
  big_val, big_idx = torch.topk(preds.data, k=3, dim=1)

  for x in range(len(big_idx[0])):
    sentiment.append(list(labels_dict.keys())[list(labels_dict.values()).index(big_idx[0][x].item())])

  return sentiment

def get_translation(user_text):
    lang = TextBlob(user_text).detect_language()
    if lang == "zh-CN": # Chinese
        df= pd.DataFrame([user_text])
        corpus = list(df[0].values)
        translated = model_zh.generate(**tokenizer_zh.prepare_seq2seq_batch(corpus, return_tensors='pt'))
        translated_text= [tokenizer_zh.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0], "Chinese"
    elif lang == "fr": # French
        df= pd.DataFrame([user_text])
        corpus = list(df[0].values)
        translated = model_fr.generate(**tokenizer_fr.prepare_seq2seq_batch(corpus, return_tensors='pt'))
        translated_text= [tokenizer_fr.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0], "French"
    elif lang == "fi": # Finnish
        df= pd.DataFrame([user_text])
        corpus = list(df[0].values)
        translated = model_fi.generate(**tokenizer_fi.prepare_seq2seq_batch(corpus, return_tensors='pt'))
        translated_text= [tokenizer_fi.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0], "Finnish"
    elif lang == "de": # German
        df= pd.DataFrame([user_text])
        corpus = list(df[0].values)
        
        translated = model_de.generate(**tokenizer_de.prepare_seq2seq_batch(corpus, return_tensors='pt'))
        
        translated_text= [
            tokenizer_de.decode(t, skip_special_tokens=True) for t in translated
        ]
        return translated_text[0], "German"
    elif lang == "it": # Italian
        df= pd.DataFrame([user_text])
        corpus = list(df[0].values)
        translated = model_it.generate(**tokenizer_it.prepare_seq2seq_batch(corpus, return_tensors='pt'))
        translated_text= [tokenizer_it.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0], "Italian"
    elif lang == "ja": # Japanese
        df= pd.DataFrame([user_text])
        corpus = list(df[0].values)
        translated = model_ja.generate(**tokenizer_ja.prepare_seq2seq_batch(corpus, return_tensors='pt'))
        translated_text= [tokenizer_ja.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0], "Japanese"
    elif lang == "ko": # Korean
        df= pd.DataFrame([user_text])
        corpus = list(df[0].values)
        translated = model_ko.generate(**tokenizer_ko.prepare_seq2seq_batch(corpus, return_tensors='pt'))
        translated_text= [tokenizer_ko.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0], "Korean"
    elif lang == "ms": # Malay/Indonesian
        df= pd.DataFrame([user_text])
        corpus = list(df[0].values)
        translated = model_id.generate(**tokenizer_zh.prepare_seq2seq_batch(corpus, return_tensors='pt'))
        translated_text= [tokenizer_id.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0], "Malay/Indonesian"
    elif lang == "es": # Spanish
        df= pd.DataFrame([user_text])
        corpus = list(df[0].values)
        translated = model_es.generate(**tokenizer_es.prepare_seq2seq_batch(corpus, return_tensors='pt'))
        translated_text= [tokenizer_es.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0], "Spanish"
    else: # English
        return user_text, "English"

############################################################
# Start app
############################################################
# Load emotion model
model_emotion = torch.load('model/roberta_sent.bin', map_location=torch.device('cpu'))
tokenizer_emotion = RobertaTokenizer.from_pretrained("model/")

# Load language models
config_zh = MarianConfig.from_json_file('lang\opus-mt-zh-en\config.json')
model_zh = MarianMTModel.from_pretrained("lang\opus-mt-zh-en\pytorch_model.bin", config=config_zh)
tokenizer_zh = MarianTokenizer.from_pretrained("lang\opus-mt-zh-en")

config_fr = MarianConfig.from_json_file('lang\opus-mt-fr-en\config.json')
model_fr = MarianMTModel.from_pretrained("lang\opus-mt-fr-en\pytorch_model.bin", config=config_fr)
tokenizer_fr = MarianTokenizer.from_pretrained("lang\opus-mt-fr-en")

config_fi = MarianConfig.from_json_file('lang\opus-mt-fi-en\config.json')
model_fi = MarianMTModel.from_pretrained("lang\opus-mt-fi-en\pytorch_model.bin", config=config_fi)
tokenizer_fi = MarianTokenizer.from_pretrained("lang\opus-mt-fi-en")

config_de = MarianConfig.from_json_file('lang\opus-mt-de-en\config.json')
model_de = MarianMTModel.from_pretrained("lang\opus-mt-de-en\pytorch_model.bin", config=config_de)
tokenizer_de = MarianTokenizer.from_pretrained("lang\opus-mt-de-en")

config_it = MarianConfig.from_json_file('lang\opus-mt-it-en\config.json')
model_it = MarianMTModel.from_pretrained("lang\opus-mt-it-en\pytorch_model.bin", config=config_it)
tokenizer_it = MarianTokenizer.from_pretrained("lang\opus-mt-it-en")

config_ja = MarianConfig.from_json_file('lang\opus-mt-ja-en\config.json')
model_ja = MarianMTModel.from_pretrained("lang\opus-mt-ja-en\pytorch_model.bin", config=config_ja)
tokenizer_ja = MarianTokenizer.from_pretrained("lang\opus-mt-ja-en")

config_ko = MarianConfig.from_json_file('lang\opus-mt-ko-en\config.json')
model_ko = MarianMTModel.from_pretrained("lang\opus-mt-ko-en\pytorch_model.bin", config=config_ko)
tokenizer_ko = MarianTokenizer.from_pretrained("lang\opus-mt-ko-en")

config_id = MarianConfig.from_json_file('lang\opus-mt-id-en\config.json')
model_id = MarianMTModel.from_pretrained("lang\opus-mt-id-en\pytorch_model.bin", config=config_id)
tokenizer_id = MarianTokenizer.from_pretrained("lang\opus-mt-id-en")

config_es = MarianConfig.from_json_file('lang\opus-mt-es-en\config.json')
model_es = MarianMTModel.from_pretrained("lang\opus-mt-es-en\pytorch_model.bin", config=config_es)
tokenizer_es = MarianTokenizer.from_pretrained("lang\opus-mt-es-en")

# Set emotion labels
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

# Set up Flask
app = Flask(__name__)
app.secret_key = 'the random string'

@app.route('/')
def my_form():
    return render_template('from_ex.html')

@app.route('/results', methods=['POST','GET'])
def my_form_post():
    try:
        # Enter text into input
        user_input = request.form['name_input']
        # user_input = 'missed the bus.... shit!'

        # Translate non English text input to English
        translated_input, language = get_translation(user_input)

        # Do data processing and sentiment prediction
        sentiment = get_sentiment(str(translated_input))

        # Display the results on the web after successful processing
        flash ("Text input: " + user_input)
        if language != "English":
            flash("Translated input from " + language + ": " + translated_input)
        flash ("Emotions prediction: " +str(sentiment))
    except:
        # Display the error processing if processing fails
        flash("Error, please trpe a new text with alpha characters only and character count >2.")
    finally:
        # Display on the web page
        return render_template("from_ex.html")

if __name__ == '__main__':
     app.run()
