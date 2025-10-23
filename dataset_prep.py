import pandas as pd
from sklearn.model_selection import train_test_split
import os

current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'aslg_pc12.csv')

df = pd.read_csv(data_path)

english_texts = df['text'].tolist()
gloss_texts = df['gloss'].tolist()

# define split sizes
test_size = 0.2
val_size = 0.1

def prepare_dataset():

    X_temp, X_test, y_temp, y_test = train_test_split(
        english_texts,
        gloss_texts,
        test_size=test_size,
        random_state=42
    )

    val_fraction = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        random_state=42
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }
