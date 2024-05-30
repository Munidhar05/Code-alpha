!pip install music21 tensorflow numpy



import music21
import numpy as np
import glob


def get_notes():
    notes = []
    for file in glob.glob("/content/4bros.mid"):
        midi = music21.converter.parse(file)
        notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, music21.note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, music21.chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

notes = get_notes()



unique_notes = sorted(set(notes))
note_to_int = {note: num for num, note in enumerate(unique_notes)}

sequence_length = 100
network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

# Reshape the input into a format compatible with LSTM layers
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(len(unique_notes))
network_output = np.eye(len(unique_notes))[network_output]




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation

model = Sequential([
    LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dropout(0.3),
    Dense(len(unique_notes)),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()






epochs = 100
batch_size = 64


model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size)


import random


start = np.random.randint(0, len(network_input) - 1)
pattern = network_input[start]
prediction_output = []


for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(len(unique_notes))

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = unique_notes[index]
    prediction_output.append(result)

    pattern = np.append(pattern, index)
    pattern = pattern[1:len(pattern)]
   
offset = 0
output_notes = []

for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        chord_notes = pattern.split('.')
        chord_notes = [int(note) for note in chord_notes]
        chord = music21.chord.Chord(chord_notes)
        chord.offset = offset
        output_notes.append(chord)
    else:
        note = music21.note.Note(pattern)
        note.offset = offset
        output_notes.append(note)
    offset += 0.5

midi_stream = music21.stream.Stream(output_notes)
midi_stream.write('midi', fp='output.mid')

