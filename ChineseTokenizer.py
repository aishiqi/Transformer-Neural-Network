chinese_file = './dataset/en_zh/chinese.zh' # replace this path with appropriate one
english_file = './dataset/en_zh/english.en' # replace this path with appropriate one


with open(english_file, 'r', encoding='utf-8') as file:
    english_sentences = file.readlines()
with open(chinese_file, 'r', encoding='utf-8') as file:
    chinese_sentences = file.readlines()

# Limit Number of sentences
# TOTAL_SENTENCES = 200000
# english_sentences = english_sentences[:TOTAL_SENTENCES]
# chinese_sentences = chinese_sentences[:TOTAL_SENTENCES]

english_sentences = [sentence.rstrip('\n').lower() for sentence in english_sentences]
chinese_sentences = [sentence.rstrip('\n') for sentence in chinese_sentences]


def build_vacab(sentences):
    vacab = {}
    for index in range(len(sentences)):
        chinese_sentence = sentences[index]
        for word in chinese_sentence:
            if word not in vacab:
                vacab[word] = len(vacab)
    return vacab
