from data_loading import load_data
from preprocessing import preprocess_data
from train import train_model


def main():
    df = load_data()
    X_train, X_test, Y_train, Y_test = preprocess_data(df)
    train_model(X_train, Y_train,X_test, Y_test)

    # You can add more code here to process the data or train a model


if __name__ == "__main__":
    main()