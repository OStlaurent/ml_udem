import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# Transform class name to int
labels_to_int = {np.bytes_(b'sink'): 0,
                 np.bytes_(b'moustache') : 1,
                 np.bytes_(b'nose') : 2,
                 np.bytes_(b'skateboard') : 3,
                 np.bytes_(b'penguin'): 4,
                 np.bytes_(b'peanut'): 5,
                 np.bytes_(b'skull'): 6,
                 np.bytes_(b'panda'): 7,
                 np.bytes_(b'paintbrush'): 8,
                 np.bytes_(b'nail'): 9,
                 np.bytes_(b'apple'): 10,
                 np.bytes_(b'rifle'): 11,
                 np.bytes_(b'mug'): 12,
                 np.bytes_(b'sailboat'): 13,
                 np.bytes_(b'pineapple'):14,
                 np.bytes_(b'spoon'): 15,
                 np.bytes_(b'rabbit'): 16,
                 np.bytes_(b'shovel'): 17,
                 np.bytes_(b'screwdriver'): 18,
                 np.bytes_(b'scorpion'): 19,
                 np.bytes_(b'rhinoceros'): 20,
                 np.bytes_(b'rollerskates'): 21,
                 np.bytes_(b'pool'): 22,
                 np.bytes_(b'octagon'): 23,
                 np.bytes_(b'pillow'):24,
                 np.bytes_(b'parrot'):25,
                 np.bytes_(b'squiggle'):26,
                 np.bytes_(b'mouth'):27,
                 np.bytes_(b'empty'):28,
                 np.bytes_(b'pencil'):29,
                 np.bytes_(b'pear'): 30
                 }

# Transform class int to string
int_to_string = {0: 'sink',
                 1: 'moustache',
                 2: 'nose',
                 3: 'skateboard',
                 4: 'penguin',
                 5: 'peanut',
                 6: 'skull',
                 7: 'panda',
                 8: 'paintbrush',
                 9: 'nail',
                 10: 'apple',
                 11: 'rifle',
                 12: 'mug',
                 13: 'sailboat',
                 14: 'pineapple',
                 15: 'spoon',
                 16: 'rabbit',
                 17: 'shovel',
                 18: 'screwdriver',
                 19: 'scorpion',
                 20: 'rhinoceros',
                 21: 'rollerskates',
                 22: 'pool',
                 23: 'octagon',
                 24: 'pillow',
                 25: 'parrot',
                 26: 'squiggle',
                 27: 'mouth',
                 28: 'empty',
                 29: 'pencil',
                 30: 'pear'
                 }

X_train = np.load('X_train_clean_bin_140_45.npy', encoding='latin1')
X_test = np.load('X_test_clean_bin_140_45.npy', encoding='latin1')

train_labels = np.genfromtxt('input/train_labels.csv', names=True, delimiter=',', dtype=[('Id', 'i8'), ('Category', 'S14')])
y = np.array([labels_to_int[e[1]] for e in train_labels])


print(X_train.shape)
print(type(X_train[0]))

print(X_test.shape)
print(X_test[0].shape)

svm_linear = SVC(kernel="linear")
svm_linear.fit(X_train,y)

result = svm_linear.predict(X_test)

#save result so we can reuse it without the computation
result.dump('svm_result.npy')

#Create the csv of prediction on the test set
id = np.array(['Id'] + [i for i in range(len(result))])
result_string = [int_to_string[e] for e in result]
result_string = ['Category'] + result_string

np.savetxt('svm_submission.csv', [e for e in zip(id, result_string)], delimiter=',', fmt='%s')
