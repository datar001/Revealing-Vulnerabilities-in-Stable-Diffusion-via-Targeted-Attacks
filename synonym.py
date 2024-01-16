import torch.nn as nn
import torch
import re, os
import pickle
import pdb

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def is_english(text):
    # matching English words using the regular expression
    pattern = re.compile(r'[^a-zA-Z\s]')
    if pattern.search(text):
        return False
    else:
        return True

def get_token_english_mask(vocab_encoder, device):
    token_num = len(vocab_encoder)
    mask = torch.ones(token_num).bool()
    for token, id in vocab_encoder.items():
        if is_english(token.replace("</w>", "")):
            mask[id] = 0
    mask = mask.to(device)
    return mask


class Synonym(nn.Module):
    def __init__(self, word_path, device) -> None:
        super(Synonym, self).__init__()
        word2id = pickle.load(open(os.path.join(word_path, "word2id.pkl"), "rb"))
        wordvec = pickle.load(open(os.path.join(word_path, "wordvec.pkl"), "rb"))
        self.word2id = word2id
        self.id2word = {id_:word_ for word_, id_ in self.word2id.items()}
        self.embedding = torch.from_numpy(wordvec)
        # normalization
        self.embedding = self.embedding / self.embedding.norm(dim=1, keepdim=True)
        self.embedding = self.embedding.to(device)
        # delete non english words
        self.delete_non_english()

    def delete_non_english(self):
        for word, id in self.word2id.items():
            if not is_english(word):
                self.embedding[id] = 0

    def transform(self, word, token_unk):
        if word in self.word2id:
            return self.embedding[self.word2id[word]]
        else:
            if isinstance(token_unk, int):
                return self.embedding[token_unk]
            else:
                return self.embedding[self.word2id[token_unk]]

    def get_synonym(self, words, k=5, word2id=None, embedding=None, id2word=None):

        word2id = word2id if word2id is not None else self.word2id
        embedding = embedding if embedding is not None else self.embedding
        id2word = id2word if id2word is not None else self.id2word

        if type(words) == str:
            words = [words]
        results = []
        for word in words:
            if len(word.split()) > 1:
                results.extend(self.get_synonym(word.split(), k=k))
            else:
                if word not in word2id:
                    results.append([word, -1, -1])
                    continue
                word_id = word2id[word]
                word_embedding = embedding[word_id]
                sims = torch.mm(word_embedding.view(1,-1), embedding.t())
                top_k_values, top_k_id = torch.topk(sims, k=k, dim=1, largest=True, sorted=False)

                for id, sim in sorted([[id.item(), value.item()] for id, value in zip(top_k_id[0], top_k_values[0])],
                                      key=lambda x:x[1], reverse=True):
                    cur_word = id2word[id]
                    if cur_word != word:
                        results.append([cur_word, id, sim])
        return results

    def get_synonym_by_tokenizer(self, word, tokenizer, k=5):
        embedding = tokenizer.token_embedding.weight
        embedding = embedding / embedding.norm(dim=1, keepdim=True)

        word2id = tokenizer.encoder
        id2word = tokenizer.decoder

        return self.get_synonym(word, k, word2id=word2id, embedding=embedding, id2word=id2word)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    synonym_model = Synonym(word_path="./Word2Vec/", device=device)
    pdb.set_trace()
    synonym_words = synonym_model.get_synonym(["ice cream"], k=10)
    forbidden_words = ["ice cream"]
    forbidden_words.extend([word[0] for word in synonym_words])
    print(forbidden_words)

