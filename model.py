import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

df = pd.read_csv("combined_data.csv")

x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV')
tokenizer.fit_ont_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

max_len = 20
x_train_pad = pad_sequences(x_train_seq, maxlen=max_len, padding='post')
x_test_pad = pad_sequences(x_test_seq, maxlen=max_len, padding='post')

model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_len),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train_pad, y_train, epochs=10, batch_size=4, validation_data=(x_test_pad, y_test))

def predict(text):
    seq = tokenizer.test_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded)[0][0]
    if prediction > 0.5:
        return "spam"
    else:
        return "not spam"

import gradio as gr

interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, placeholder="Enter a message here..."),
    outputs="text",
    title="Spam Message Classifier",
    description="Enter a message and the model will predict whether it's Spam or Not Spam."
)

interface.launch()