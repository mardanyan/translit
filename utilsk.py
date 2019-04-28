from keras.layers import Dense, Input, LSTM, Concatenate, Bidirectional
from keras.models import Model

def define_model(hdim, depth, trans_vocab_size=0, vocab_size=0, is_train = False):
    input_input = Input(shape=(None, trans_vocab_size))
    layer = input_input

    for _ in range(depth):
        layer = Bidirectional(LSTM(hdim, return_sequences=True, return_state=False))(layer)
        layer = Dense(hdim)(layer)

    layer = Bidirectional(LSTM(hdim, return_sequences=True, return_state=False))(layer)
    layer = Dense(hdim)(layer)
    layer = Concatenate()([layer, input_input])
    layer = Dense(vocab_size, activation='softmax')(layer)

    model = Model(inputs=input_input, outputs=layer)

    model.compile(optimizer="adam", # rmsprop,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model
