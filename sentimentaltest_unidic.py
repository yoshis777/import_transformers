from transformers import pipeline 
from transformers import AutoModelForSequenceClassification 
from transformers import BertJapaneseTokenizer 
 
TARGET_TEXT = "誰でもできる感情分析です。簡単であるので、気軽に試してみましょう。"
 
model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment') 
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', 
    mecab_kwargs={"mecab_dic": 'unidic', "mecab_option": None}) 
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) 
 
print(nlp(TARGET_TEXT))

TARGET_TEXTS = [
    '嬉しいです',
    '悲しいです',
    '嬉しくて悲しいです',
    '悲しくて嬉しいです',
    '悲しいけど、嬉しいです',
    '嬉しいけど、悲しいです',
]
for text in TARGET_TEXTS:
    print('| ' + text + ' | ' + str(nlp(text)) + ' |')
