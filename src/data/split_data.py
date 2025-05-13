from sklearn.model_selection import train_test_split

def split_data(df, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits the DataFrame into train, validation, and test sets.
    """

    X_temp, X_test, y_temp, y_test = train_test_split(df, y, test_size=test_size, random_state=random_state)

    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__" :
    print("x")

