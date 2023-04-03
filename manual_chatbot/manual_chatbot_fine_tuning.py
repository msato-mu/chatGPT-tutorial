import openai
import os
import pandas as pd
import MeCab
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import EncoderDecoderModel
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

openai.api_key = os.environ.get("OPENAI_API_KEY")


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_urls(text):
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_non_alphanumeric(text):
    text = re.sub(r'\W+', ' ', text)
    return text.strip()


def clean_text(text):
    # テキストのクリーニング処理を実装
    remove_emoji(text)
    remove_html_tags(text)
    remove_non_alphanumeric(text)
    remove_urls(text)

    return text


def tokenize(text):
    mecab = MeCab.Tagger('-Owakati')
    return mecab.parse(text).strip()


def tokenize_data(example):
    return tokenizer(example['質問'], example['回答'], truncation=True, padding='max_length', max_length=512)


def generate_answer(question):
    input_ids = tokenizer.encode(question, return_tensors="pt")
    output_ids = model.generate(input_ids)
    answer = tokenizer.decode(output_ids[0])
    return answer


# CSVファイルを読み込み
data = pd.read_csv('../data/manual_data.csv')

# テキストデータを前処理
data['質問'] = data['質問'].apply(clean_text).apply(tokenize)
data['回答'] = data['回答'].apply(clean_text).apply(tokenize)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, validation_data = train_test_split(
    train_data, test_size=0.25, random_state=42)

# データセットを'datasets'ライブラリ形式に変換
train_dataset = Dataset.from_pandas(train_data)
validation_dataset = Dataset.from_pandas(validation_data)

# トークナイザ（Tokenizer）のインスタンスを生成
model_checkpoint = "cl-tohoku/bert-base-japanese-whole-word-masking"  # 任意の日本語モデルを選択
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# データセットをトークナイズ
train_dataset = train_dataset.map(tokenize_data, batched=True)
validation_dataset = validation_dataset.map(tokenize_data, batched=True)

# エンコーダーとデコーダーのチェックポイントを設定
encoder_checkpoint = model_checkpoint
decoder_checkpoint = model_checkpoint

# エンコーダーとデコーダーのモデルをインスタンス化し、EncoderDecoderModelを作成
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_checkpoint, decoder_checkpoint)

# DataCollatorをインスタンス化
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# モデルを訓練
trainer.train()

question = "製品Aの使い方を教えてください。"
answer = generate_answer(question)
print(answer)
