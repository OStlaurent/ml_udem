import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from sklearn.preprocessing import Binarizer

# # modifie le nombre de caractere possible a print dans la console
# desired_width = 320
# np.set_printoptions(linewidth=desired_width)
# np.set_printoptions(threshold=np.nan)

threshold = 140
min_size = 45

X_train = np.load('X_train.npy', encoding='latin1')
X_test = np.load('X_test.npy', encoding='latin1')

# Nouveaux datasets
X_train_clean = np.zeros(X_train.shape)
X_train_clean_bin = np.zeros(X_train.shape)

# transform pixel with value below threshold to 0 and above to 1
binarizer = Binarizer(threshold=threshold)
binarizer2 = Binarizer(threshold=0)

X_train = binarizer.transform(X_train).astype(int)
# plt.imshow(X_train[0].reshape(100,100))
# plt.show()

for i in range(X_train.shape[0]):
    #print(train_labels[i][1])

    # # show original
    # plt.imshow(X_train[i].reshape(100, 100))
    # plt.show()

    # # # Tentative d'ajouter du preprocessing suplementaire (mais moins bon)
    # # X_train[i] = morphology.erosion(X_train[i].reshape(100,100)).reshape(10000)
    # # plt.imshow(X_train[i].reshape(100,100))
    # # plt.show()

    # Two pixels are connected when they are neighbors and have the same value
    # Labeled array, where all connected regions are assigned the same integer value.
    x_i_label = morphology.label(X_train[i].reshape(100, 100), connectivity=2)
    # plt.imshow(x_i_label)
    # plt.show()

    # Expects an array with labeled objects, and removes objects smaller than min_size
    x_i_clean = morphology.remove_small_objects(x_i_label, min_size=min_size, connectivity=2)
    # # show apres binarisation + removing small object
    # plt.imshow(x_i_clean.reshape(100,100))
    # plt.show()

    # # binarize result
    x_i_clean_bin = binarizer2.transform(x_i_clean, copy=True)
    # plt.imshow(x_i_clean_bin.reshape(100,100))
    # plt.show()

    X_train_clean[i] = x_i_clean.reshape(10000,)
    X_train_clean_bin[i] = x_i_clean_bin.reshape(10000,)


X_train_clean.dump("X_train_clean_"+str(threshold)+"_"+str(min_size)+".npy")
X_train_clean_bin.dump("X_train_clean_bin_"+str(threshold)+"_"+str(min_size)+".npy")


# les variables qui modifient beaucoup les resultats sont min_size dans remove_small_objects() et threshold dans Binarize.
# Une min_size trop grande et on enleve les petits dessins, trop petite on laisse du bruit

# S'il y a plusieurs formes restantes apres les transformations, les differentes forment sont de differentes couleur
# (label dessine la premiere forme avec une seule valeure (int), la deuxieme forme est dessiné avec une valeur differente, et etc)
# Alors, entre X_train[i] et X_train[j], les valeurs utilisé pour les pixels sont pas les mêmes.
# Au cas ou cela affecte le classificateur, il y a X_train_clean_bin qui a été de nouveau binarizer à la fin.


## Meme chose pour X_test

# Nouveaux datasets
X_test_clean = np.zeros(X_test.shape)
X_test_clean_bin = np.zeros(X_test.shape)

X_test = binarizer.transform(X_test).astype(int)

for i in range(X_test.shape[0]):

    # Two pixels are connected when they are neighbors and have the same value
    # Labeled array, where all connected regions are assigned the same integer value.
    x_i_label = morphology.label(X_test[i].reshape(100, 100), connectivity=2)

    # Expects an array with labeled objects, and removes objects smaller than min_size
    x_i_clean = morphology.remove_small_objects(x_i_label, min_size=min_size, connectivity=2)

    # # binarize result
    x_i_clean_bin = binarizer2.transform(x_i_clean, copy=True)

    X_test_clean[i] = x_i_clean.reshape(10000,)
    X_test_clean_bin[i] = x_i_clean_bin.reshape(10000,)


X_test_clean.dump("X_test_clean_"+str(threshold)+"_"+str(min_size)+".npy")
X_test_clean_bin.dump("X_test_clean_bin_"+str(threshold)+"_"+str(min_size)+".npy")