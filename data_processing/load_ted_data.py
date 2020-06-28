from .ted_reader import MultiLingualAlignedCorpusReader
import os

ted_data_path = "/data/rrjin/corpus_data/raw_ted_corpus"


def load(src_lang, trg_lang):

    output_data_path = "/data/rrjin/corpus_data/ted_data/{}_{}".format(src_lang, trg_lang)

    train_lang_dict={'source': [src_lang], 'target': [trg_lang]}
    eval_lang_dict = {'source': [src_lang], 'target': [trg_lang]}

    obj = MultiLingualAlignedCorpusReader(corpus_path=ted_data_path,
                                          lang_dict=train_lang_dict,
                                          target_token=True,
                                          corpus_type='file',
                                          eval_lang_dict=eval_lang_dict,
                                          zero_shot=False,
                                          bilingual=True)

    os.makedirs(output_data_path, exist_ok=True)
    obj.save_file(output_data_path + "/train.{}".format(src_lang),
                  split_type='train', data_type='source')
    obj.save_file(output_data_path + "/train.{}".format(trg_lang),
                  split_type='train', data_type='target')

    obj.save_file(output_data_path + "/test.{}".format(src_lang),
                  split_type='test', data_type='source')
    obj.save_file(output_data_path + "/test.{}".format(trg_lang),
                  split_type='test', data_type='target')

    obj.save_file(output_data_path + "/dev.{}".format(src_lang),
                  split_type='dev', data_type='source')
    obj.save_file(output_data_path + "/dev.{}".format(trg_lang),
                  split_type='dev', data_type='target')


# src_lang_name = ["es", "pt-br", "fr", "ru", "he", "ar", "ko", "zh-cn", "it", "ja", "zh-tw", "nl", "ro", "tr", "devi", "pl", "pt", "bg", "el", "fa", "sr", "hu", "hr", "uk", "cs", "id", "th",
#                  "sv", "sk", "sq", "lt", "dacalv", "my", "sl", "mk", "fr-ca", "fi", "hy", "hi", "nb", "ka", "mn", "et", "ku", "gl", "mr", "zh", "ur", "eoms", "az", "ta", "bn", "kk", "be", "eu", "bs"]


# src_lang_name = ["pl", "pt", "bg", "el", "fa", "sr", "hu", "hr", "uk", "cs", "id", "th", "sv", "sk", "sq", "lt", "dacalv", "my", "sl",
#                  "mk", "fr-ca", "fi", "hy", "hi", "nb", "ka", "mn", "et", "ku", "gl", "mr", "zh", "ur", "eoms", "az", "ta", "bn", "kk", "be", "eu", "bs"]

# src_lang_name = ["my", "sl", "mk", "fr-ca", "fi", "hy", "hi", "nb", "ka", "mn", "et",
#                  "ku", "gl", "mr", "zh", "ur", "eoms", "az", "ta", "bn", "kk", "be", "eu", "bs"]


src_lang_name = ["az", "ta", "bn", "kk", "be", "eu", "bs"]


for lang in src_lang_name:
    load(lang, "en")
