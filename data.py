import pickle


def get_test_data_from_pkl(test_data_path):
    with open(test_data_path, "rb") as f:
        data = pickle.load(f)
    return data
