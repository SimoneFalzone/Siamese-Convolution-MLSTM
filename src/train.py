"""
This module shall handle the training of models, including the respective transformation of data to concur with the
guideline for the utilized framework

"""
import numpy
import pandas
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding
from cnn_mlstm import CNN_MLSTM
from tensorflow.keras.models import model_from_json
from keras_lr_finder.lr_finder import LRFinder
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


def load_dataset(path_to_data: str = ''):
    """
    Loads any JSON exported from pandas.DataFrame
    :param path_to_data: str (Path to JSON pandas.DataFrame)
    :return: pandas.DataFrame
    """
    if path_to_data is None or len(str(path_to_data)) < 1:
        return -1
    path_to_data = Path(path_to_data)
    if not path_to_data.exists() or not path_to_data.is_file() or '.json' not in path_to_data.parts[-1]:
        return -1

    return pandas.read_json(str(path_to_data))


def preprocess_dataset(dataset: pandas.DataFrame, to_lower: bool = True):
    """
    Do relevant preprocessing to the dataset
    :param dataset: pandas.DataFrame (dataset)
    :param to_lower: bool (shall the sentence-content transformed to lowercase)
    :return: -1 if error, else None
    """
    if not isinstance(dataset, pandas.DataFrame):
        return -1

    if to_lower:
        dataset['sentence_A'] = dataset['sentence_A'].str.lower()
        dataset['sentence_B'] = dataset['sentence_B'].str.lower()


def build_vocabulary(alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"):
    """
    Builds vocabulary based on alphabet
    Args:
        alphabet: is of string. Each character relevant to the alphabet is concatenated

    Returns:
        Dictionary containing the vocabulary. In addition an incremental index is stored per character
    """
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1

    return char_dict


def init_tokenizer(char_dict: dict):
    """
    Creates the tokenizer based on the passed vocabulary. Additionally an UKNOWN-token is added for it to later represent
    any unknown character. This in generally known as out-of-vocabulary error.
    Args:
        char_dict: Alphabet with incremental index per entry

    Returns:
        Tokenizer
    """
    # Init tokenizer with "UNK" (unknown) token for out-of-vocabulary
    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    # Use char_dict to replace the tk.word_index
    tokenizer.word_index = char_dict.copy()
    # Add 'UNK' to the vocabulary
    tokenizer.word_index[tokenizer.oov_token] = max(char_dict.values()) + 1

    return tokenizer


def train_val_test_split(dataset: pandas.DataFrame):
    """
    This is a helper function for performing a train, val and test split. This is done by using the train_test_split
    twice. First, on the entire dataset the seperate the test split. Second, on the remaining data to create a train and
    a val set.
    Args:
        dataset: pandas.DataFrame. This dataframe contains the input data

    Returns:
        3-tuple of (train, val, test) data. Type is pandas.DataFrame

    """
    rest, test = train_test_split(dataset, test_size=0.25, stratify=dataset['relatedness_score'])
    train, val = train_test_split(rest, test_size=0.1, stratify=rest['relatedness_score'])

    return train, val, test


def make_onehot_embedding_weights(tokenizer: Tokenizer):
    """
    Creates onehot encoded weight matrix for the embedding.
    Args:
        tokenizer: tokenizer contains the vocabulary on which the embedding is build on

    Returns:
        numpy.array containing the embedding matrix
    """
    if tokenizer is None:
        return -1
    vocab_size = len(tokenizer.word_index)

    # init embedding matrix
    embedding_weights = []
    embedding_weights.append(numpy.zeros(vocab_size))

    # make one-hot encoding for each letter in the alphabet
    for char, i in tokenizer.word_index.items():
        onehot = numpy.zeros(vocab_size)
        onehot[i - 1] = 1
        embedding_weights.append(onehot)
    return numpy.array(embedding_weights)


def convert_text_to_sequence(tokenizer: Tokenizer, text: list, sequence_length=128, padding='post', padding_value=0):
    """
    This function converts a list of sentences into a list containing a sequence by utilizing the vocabulary inside
    the tokenizer.
    Args:
        tokenizer: Tokenizer containing the vocabulary
        text: List of strings. One sentence per string.
        sequence_length: Desired output length of the sequence. Longer sequences get truncated. Shorter sequences get
        extended according to PADDING and PADDING_VALUE argument.
        padding: Whether padding shall be performed by appending PADDING_VALUE to the front or back of the sequence.
        padding_value: The value with which shorter sequences shall be filled

    Returns:
        List of sequences
    """
    if tokenizer is None or text is None or len(text) < 1:
        return -1
    # Convert string to index
    text_sequences = tokenizer.texts_to_sequences(text)

    # Padding
    text_data = pad_sequences(text_sequences, maxlen=sequence_length, padding=padding, value=padding_value)

    # Convert to numpy array
    return numpy.array(text_data, dtype='float32')


def load_from_json(path: str = './model_save/', filename: str = "model"):
    """
    This is a helper function for loading a model from JSON. KERAS does not load the model and it's weights in one go.
    Therefore this function was created to ease to loading of a model
    Args:
        path: String containing the path to the folder where the model_save is stored
        filename: name of the JSON and H5 model files

    Returns:
        some KERAS model
    """
    # load model architecture
    json_file = open(path + filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load weights
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path + filename + '.h5')

    return loaded_model


def find_learningrate(model, train_x, train_y, start_lr=0.0001, end_lr=1, moving_average=20,
                      loss_function=losses.MeanAbsoluteError(), metric=metrics.MeanAbsoluteError(), no_epochs=5,
                      batch_size=32):
    """
    Utilizes KERAS-LR-Finder for finding a better learning rate for fixed learning rate optimization algorithms.
    Source: https://pypi.org/project/keras-lr-finder/
    Args:
        model: KERAS.model
        train_x: train features
        train_y: train labels
        start_lr: initial learning rate
        end_lr: limit of learning rate
        moving_average: smoothing of learning rate curve. Otherwise it is to jagged
        loss_function: loss function to evaluate model training on
        metric: metric to evaluate model on
        no_epochs: number of training epochs per algorithm
        batch_size: batch size of training data for the model training process

    Returns:
        dict containing the search result. Results will also be printed on the fly
    """
    # Determine tests you want to perform
    tests = [
        (optimizers.SGD(), 'SGD optimizer'),
        (optimizers.Adam(), 'Adam optimizer'),
        (optimizers.Adadelta(), 'Adadelta optimizer'),
        (optimizers.RMSprop(), 'RMSprop optimizer'),
    ]
    # Set containers for tests
    test_learning_rates = []
    test_losses = []
    test_loss_changes = []
    labels = []

    # Perform each test
    for test_optimizer, label in tests:

        # Compile the model
        model.compile(loss=loss_function,
                      optimizer=test_optimizer,
                      metrics=[metric])

        # Instantiate the Learning Rate Range Test / LR Finder
        lr_finder = LRFinder(model)

        # Perform the Learning Rate Range Test
        outputs = lr_finder.find(train_x, train_y, start_lr=start_lr, end_lr=end_lr, batch_size=batch_size,
                                 epochs=no_epochs)

        # Get values
        learning_rates = lr_finder.lrs
        losses = lr_finder.losses
        loss_changes = []

        # Compute smoothed loss changes
        # Inspired by Keras LR Finder: https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py
        for i in range(moving_average, len(learning_rates)):
            loss_changes.append((losses[i] - losses[i - moving_average]) / moving_average)

        # Append values to container
        test_learning_rates.append(learning_rates)
        test_losses.append(losses)
        test_loss_changes.append(loss_changes)
        labels.append(label)

    # Generate plot for Loss Deltas
    for i in range(0, len(test_learning_rates)):
        plt.plot(test_learning_rates[i][moving_average:], test_loss_changes[i], label=labels[i])
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.ylabel('loss delta')
    plt.xlabel('learning rate (log scale)')
    plt.title('Results for Learning Rate Range Test / Loss Deltas for Learning Rate')
    plt.savefig('loss_deltas.png')
    plt.show()

    # Generate plot for Loss Values
    for i in range(0, len(test_learning_rates)):
        plt.plot(test_learning_rates[i], test_losses[i], label=labels[i])
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('learning rate (log scale)')
    plt.title('Results for Learning Rate Range Test / Loss Values for Learning Rate')
    plt.savefig('loss_values.png')
    plt.show()

    tmp = {'test_learning_rates': test_learning_rates, 'test_loss_changes': test_loss_changes, 'labels': labels,
           'test_losses': test_losses}

    save_obj(obj=tmp, name='hyperparam_eval')

    return tmp


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def construct_sentence(datarow=None, only_number: bool = False):
    """
    Internal helper function. Constructs a "sentece" from a datarow
    :param datarow: pandas.Series
    :return: str
    """
    if datarow is None:
        return -1

    # init relevant keys for parts and clientparts table
    parts_tablekeys = ['number', 'type', 'other']
    client_tablekeys = ['oem_info_supplier.number', 'internal_info_supplier.name']
    if only_number:
        parts_tablekeys = ['number']
    relevant_keys = []

    # check for table keys
    if all([key in datarow.keys() for key in parts_tablekeys]):
        relevant_keys = parts_tablekeys
    elif all([key in datarow.keys() for key in client_tablekeys]):
        relevant_keys = client_tablekeys
    else:
        return -1

    # create sentence
    sentence = ''
    first_append = True
    for key in relevant_keys:
        # check for some weird inputs
        try:
            row_content = str(datarow[key].values[0]) if datarow[key].values[0] is not numpy.nan else ''
        except:
            try:
                row_content = str(datarow[key]) if datarow[key] is not numpy.nan else ''
            except:
                raise
        if len(row_content) <= 0:
            continue
        # create sentence piece by piece
        if first_append:
            sentence += row_content
            first_append = False
        elif not first_append:
            sentence = sentence + ' ' + row_content
    if len(sentence) == 0:
        return -1

    return sentence

if __name__ == '__main__':
    input_sequence_length = 70

    # build char dictionary
    char_dict = build_vocabulary()
    tokenizer = init_tokenizer(char_dict=char_dict)

    # Load dataset in SICK-like dataformat
    # TODO Load the dataset you are interested in and especially adjust the location of the data
    dataset = load_dataset(
        path_to_data="/data/train_data.json")

    # convert dataset to fit with alphabet
    preprocess_dataset(dataset=dataset)

    embedding_weights = make_onehot_embedding_weights(tokenizer=tokenizer)

    # create embedding layer for encoding input sequences
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1,
                                output_dim=69,
                                input_length=input_sequence_length,
                                weights=[embedding_weights])
    embedding_layer.trainable = False

    # do stratified cross validiation for model evaluation
    num_folds = 5
    skf = StratifiedKFold(n_splits=num_folds)
    acc_per_fold = []
    loss_per_fold = []

    fold_no = 1
    for train_idx, test_idx in skf.split(dataset[['sentence_A', 'sentence_B']].values,
                                         dataset['relatedness_score'].values):
        X_train, X_test = dataset[['sentence_A', 'sentence_B']].iloc[train_idx].values, \
                          dataset[['sentence_A', 'sentence_B']].iloc[test_idx].values
        y_train, y_test = dataset['relatedness_score'].iloc[train_idx].values, dataset['relatedness_score'].iloc[
            test_idx].values

        sentence_A_train = convert_text_to_sequence(tokenizer=tokenizer, text=X_train[:, 0].tolist(),
                                                    sequence_length=input_sequence_length)
        sentence_B_train = convert_text_to_sequence(tokenizer=tokenizer, text=X_train[:, 1].tolist(),
                                                    sequence_length=input_sequence_length)

        sentence_A_test = convert_text_to_sequence(tokenizer=tokenizer, text=X_test[:, 0].tolist(),
                                                   sequence_length=input_sequence_length)
        sentence_B_test = convert_text_to_sequence(tokenizer=tokenizer, text=X_test[:, 1].tolist(),
                                                   sequence_length=input_sequence_length)

        model = CNN_MLSTM(embedding_layer=embedding_layer, n_hidden_states=100,
                          input_dimension=input_sequence_length)
        model.fit(left_data=sentence_A_train, right_data=sentence_B_train, targets=y_train,
                  validation_data=None, epochs=30)
        scores = model.model.evaluate([sentence_A_test, sentence_B_test], y_test)

        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {numpy.mean(acc_per_fold)} (+- {numpy.std(acc_per_fold)})')
    print(f'> Loss: {numpy.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


