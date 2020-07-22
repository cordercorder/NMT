import pickle


def save_lang_vec(lang_vec, lang_vec_path):

    with open(lang_vec_path, "wb") as f:
        pickle.dump(lang_vec, f)


def load_lang_vec(lang_vec_path):

    with open(lang_vec_path, "rb") as f:
        lang_vec = pickle.load(f)
        return lang_vec
