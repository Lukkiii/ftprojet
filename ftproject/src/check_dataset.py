import pandas as pd

def load_data():
    try:
        df_train = pd.read_csv("../data/ftdataset_train.tsv", sep=' *\t *', encoding='utf-8', engine='python')
        df_val = pd.read_csv("../data/ftdataset_val.tsv", sep=' *\t *', encoding='utf-8', engine='python')
        df_test = pd.read_csv("../data/ftdataset_test.tsv", sep=' *\t *', encoding='utf-8', engine='python')
        return df_train, df_val, df_test
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None

def check_dataset_sizes():
    train_data, val_data, test_data = load_data()
    
    if train_data is not None:
        print("\n=== Dataset Sizes ===")
        print(f"Training set:   {len(train_data)} samples")
        print(f"Validation set: {len(val_data)} samples")
        print(f"Test set:       {len(test_data)} samples")
        print(f"Total:          {len(train_data) + len(val_data) + len(test_data)} samples")

if __name__ == "__main__":
    check_dataset_sizes()