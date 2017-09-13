import numpy as np

"""
def next_batch(imgs, labels, batch_size=32):
    
        Random shuffle and return the next batch data
        Arg:    imgs        - The whole image feature array (N * 32 * 32 * 3)
                labels      - The whole label one-hot array (N * 10)
                batch_size  - The batch size in each batch
        Ret:    The batch data about image, location and label array
    
    shuffle_index = np.arange(0, len(imgs))
    np.random.shuffle(shuffle_index)
    select_index = shuffle_index[:batch_size]
    batch_img = [ imgs[i] for i in select_index ]
    batch_label = [ labels[i] for i in select_index ]
    return np.asarray(batch_img), np.asarray(batch_label)
"""

batch_index = 0
def next_batch(imgs, labels, batch_size=32):
    global batch_index
    if batch_index + batch_size >= np.shape(imgs)[0]:
        batch_index = -1 * batch_index
    batch_index += batch_size
    return imgs[batch_index:batch_index+batch_size, :, :, :], labels[batch_index:batch_index+batch_size, :]

def to_categorical(y, num_classes=None):
    """
        The implementation of to_categorical which is defined in Keras
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical