import joblib as pickle


def create_pickle(data=None, filename=None):
    if data is not None and filename is not None:
        pickle.dump(value=data, filename=filename)
    else:
        raise ValueError("No data provided".capitalize())
