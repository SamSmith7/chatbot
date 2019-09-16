import numpy as np
from keras.models import load_model


model = load_model('./emotion-model.keras')

test = np.load('./emotion_data_test.npy')

test_x = np.stack(test[:, 0], axis=0)
test_y = np.stack(test[:, 1], axis=0)

res = model.evaluate(test_x, test_y)

print(res)
