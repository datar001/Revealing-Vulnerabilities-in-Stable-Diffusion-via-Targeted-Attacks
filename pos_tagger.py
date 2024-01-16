import nltk
import os
import torch.nn as nn
import numpy as np
import pdb


_POS_MAPPING = {
    "JJ": "adj",
    "VB": "verb",
    "NN": "noun",
    "RB": "adv",
    "IN": "prep",
    "DT": "a(n)",
}

class word_pos(nn.Module):
    def __init__(self, model_path):
        super(word_pos, self).__init__()
        self.ret = nltk.tag.PerceptronTagger(load=False)
        self.ret.load("file:" + os.path.join(model_path))


    def modify_word(self, prompt, ori_object, pad_word, replace_type):

        prompt_words = prompt.replace(ori_object, "<pad>")  # fix the ori object
        prompt_words = prompt_words.split()
        for i, (word, pos) in enumerate(self.ret.tag(prompt_words)):
            if replace_type[0] == "all":
                if word == "<pad>" or pos[:2] == "NN" or pos[:2] == "DT": # or pos[:2] == "IN"
                    continue
                else:
                    prompt_words[i] = pad_word
            else:
                if word == "<pad>" or pos[:2] not in _POS_MAPPING or _POS_MAPPING[pos[:2]] not in replace_type:
                    continue
                else:
                    prompt_words[i] = pad_word
        modified_prompt = " ".join(prompt_words)
        modified_prompt = modified_prompt.replace("<pad>", ori_object)  # recover the ori object

        return modified_prompt


if __name__ == "__main__":
    pos_model = word_pos(model_path="./perceptrontagger_model/averaged_perceptron_tagger.pickle")
    pos_model.modify_word("a photo of cat", "cat", "<start_of_text>", replace_type=["verb", "adj", "adv"])
