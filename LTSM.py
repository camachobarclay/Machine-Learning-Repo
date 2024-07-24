
# find length of passed word and make an index list
len_word = len(clean_word)
word_len_range = list(range(0,len_word))

# find indices in clean word where there are correct letters and
# those that still need to be guessed
cor_let_ind = []
cor_let_ind = [n for n in word_len_range if clean_word[n].isalpha()]
incor_let_ind = [n for n in word_len_range if not clean_word[n].isalpha()]
cor_len = len(cor_let_ind)
incor_len = len(incor_let_ind)

X_dictionary = []
y_dictionary = []

for dict_word in current_dictionary:
    if len(dict_word) != len_word:
        continue
    spliced_word_X = [(ord(word[n]) - 96) for n in ind1]
    spliced_word_y = [(ord(word[n]) - 96) for n in ind2[0]]
    X_dictionary.append(spliced_word_X)
    y_dictionary.append(spliced_word_y[0])

X = np.array(X_dictionary)
y = np.array(y_dictionary)    

print("X = ",X)
print("Y = ",y)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

model = tf.keras.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)

test_data = np.array([90, 100, 110, 120, 130])
test_data = test_data.reshape((1, n_steps, n_features))
test_data


predictNextNumber = model.predict(test_data, verbose=1)
print(predictNextNumber)
