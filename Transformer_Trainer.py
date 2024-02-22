import time
from transformer import Transformer # this is the transformer.py file
import torch
import numpy as np
import json
import os
from tqdm import tqdm
import wandb
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader

argparser = argparse.ArgumentParser()
argparser.add_argument("--checkpoint", type=str)
argparser.add_argument("--wandb_checkpoint", type=str)
argparser.add_argument("--evaluation_mode", type=bool)
args = argparser.parse_args()


run = wandb.init(
    project="Transformer-Translation",
    notes="My Transformer Translation",
    tags=["Transformer", "Translation"]
)

# english_file = './dataset/en_ch_small/train.en' # replace this path with appropriate one
# chinese_file = './dataset/en_ch_small/train.kn' # replace this path with appropriate one

chinese_file = './dataset/en_zh/chinese.zh' # replace this path with appropriate one
english_file = './dataset/en_zh/english.en' # replace this path with appropriate one

# english_file = './dataset/en_zh/chinese_small.zh' # replace this path with appropriate one
# chinese_file = './dataset/en_zh/english_small.en' # replace this path with appropriate one

# Generated this by filtering Appendix code

def build_vacab(sentences, leading_tokens = [], trailing_tokens = []):
    vacab = {}
    index_to_word = []
    for token in leading_tokens:
        vacab[token] = len(vacab)
        index_to_word.append(token)
    for index in tqdm(range(len(sentences))):
        chinese_sentence = sentences[index]
        for word in chinese_sentence:
            if word not in vacab:
                vacab[word] = len(vacab)
                index_to_word.append(word)

    for token in trailing_tokens:
        vacab[token] = len(vacab)
        index_to_word.append(token)
    return vacab, index_to_word

def load_or_build_vacab(vocab_file, sentences, leading_tokens = [], trailing_tokens = []):
    if False and os.path.exists(vocab_file):
        print("Loading vocabulary from {}".format(vocab_file))
        cache = json.load(open(vocab_file))
        vocabulary = cache["vocabulary"]
        index_to_word = cache["index_to_word"]
    else:
        print("Building vocabulary for {}".format(vocab_file))
        vocabulary, index_to_word = build_vacab(sentences, leading_tokens, trailing_tokens)
        cache = {
            "vocabulary": vocabulary,
            "index_to_word": index_to_word
        }
        open(vocab_file, "w").write(json.dumps(cache, indent=2))
    return vocabulary, index_to_word


START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

leading_tokens = [START_TOKEN]
trailing_tokens = [PADDING_TOKEN, END_TOKEN]

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

PERCENTILE = 97
print(f"{PERCENTILE}th percentile length Chinese: {np.percentile([len(x) for x in chinese_sentences], PERCENTILE)}")
print(f"{PERCENTILE}th percentile length English: {np.percentile([len(x) for x in english_sentences], PERCENTILE)}")

max_sequence_length = 200

def is_valid_length(sentence, max_sequence_length):
    return len(sentence) < (max_sequence_length - 1)  # need to re-add the end token so leaving 1 space


valid_sentence_indicies = []
for index in range(len(chinese_sentences)):
    chinese_sentence, english_sentence = chinese_sentences[index], english_sentences[index]
    if is_valid_length(chinese_sentence, max_sequence_length) \
            and is_valid_length(english_sentence, max_sequence_length):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(chinese_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

chinese_sentences = [chinese_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

# chinese_sentences[:3]

chinese_to_index, index_to_chinese = load_or_build_vacab("cache/chinese_vocabulary.txt", chinese_sentences, leading_tokens, trailing_tokens)
english_to_index, index_to_english = load_or_build_vacab("cache/english_vocabulary.txt", english_sentences, leading_tokens, trailing_tokens)


d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 2
ch_vocab_size = len(chinese_to_index)


transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          ch_vocab_size,
                          english_to_index,
                          chinese_to_index,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)




class TextDataset(Dataset):

    def __init__(self, english_sentences, chinese_sentences):
        self.english_sentences = english_sentences
        self.chinese_sentences = chinese_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.chinese_sentences[idx]


dataset = TextDataset(english_sentences, chinese_sentences)

len(dataset)

train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)


criterian = nn.CrossEntropyLoss(ignore_index=chinese_to_index[PADDING_TOKEN],
                                reduction='none')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
start_epoch = 0

checkpoint_file_path = None
if args.wandb_checkpoint:
    print(f"Loading checkpoint from wandb {args.wandb_checkpoint}")
    artifact = run.use_artifact(f'aishiqi/{run.project}/{args.wandb_checkpoint}', type='model')
    artifact_dir = artifact.download()
    checkpoint_file_path = os.path.join(artifact_dir, 'model.pth')
elif args.checkpoint:
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint_file_path = f"checkpoints/{args.checkpoint}"

if checkpoint_file_path:
    checkpoint = torch.load(checkpoint_file_path)
    transformer.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    transformer.to(device)
else:
    # When computing the loss, we are ignoring cases when the label is the padding token
    for params in transformer.parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
if checkpoint_file_path:
    optim.load_state_dict(checkpoint['optimizer_state_dict'])



NEG_INFTY = -1e9


def create_masks(eng_batch, ch_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        eng_sentence_length, ch_sentence_length = len(eng_batch[idx]), len(ch_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        ch_chars_to_padding_mask = np.arange(ch_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, ch_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, ch_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, ch_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 200

wandb.config = {
    "max_sequence_length" : max_sequence_length,
    "d_model" : d_model,
    "batch_size" : batch_size,
    "ffn_hidden" : ffn_hidden,
    "num_heads" : num_heads,
    "drop_prob" : drop_prob,
    "num_layers" : num_layers,
    "ch_vocab_size" : ch_vocab_size,
    "num_epochs" : num_epochs
}

def train():
    previous_batch_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch}")
        iterator = iter(train_loader)
        for batch_num, batch in enumerate(iterator):
            transformer.train()
            eng_batch, ch_batch = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch,
                                                                                                                  ch_batch)
            optim.zero_grad()
            ch_predictions = transformer(eng_batch,
                                         ch_batch,
                                         encoder_self_attention_mask.to(device),
                                         decoder_self_attention_mask.to(device),
                                         decoder_cross_attention_mask.to(device),
                                         enc_start_token=False,
                                         enc_end_token=False,
                                         dec_start_token=True,
                                         dec_end_token=True)
            labels = transformer.decoder.sentence_embedding.batch_tokenize(ch_batch, start_token=False, end_token=True)
            loss = criterian(
                ch_predictions.view(-1, ch_vocab_size).to(device),
                labels.view(-1).to(device)
            ).to(device)
            valid_indicies = torch.where(labels.view(-1) == chinese_to_index[PADDING_TOKEN], False, True)
            loss = loss.sum() / valid_indicies.sum()
            loss.backward()
            optim.step()
            # train_losses.append(loss.item())
            if batch_num % 100 == 0:
                delta_time = time.time() - previous_batch_time
                previous_batch_time = time.time()
                print(f"Iteration {batch_num} / {len(train_loader)}, Loss: {loss.item()}, 100 Batch Time: {delta_time}")
                print(f"English: {eng_batch[0]}")
                print(f"Chinese Translation: {ch_batch[0]}")
                wandb.log({"loss": loss})
                ch_sentence_predicted = torch.argmax(ch_predictions[0], axis=1)
                predicted_sentence = ""
                for idx in ch_sentence_predicted:
                    if idx == chinese_to_index[END_TOKEN]:
                        break
                    predicted_sentence += index_to_chinese[idx.item()]
                print(f"Chinese Prediction: {predicted_sentence}")

                transformer.eval()
                ch_sentence = ("",)
                eng_sentence = ("should we go to the mall?",)
                for word_counter in range(max_sequence_length):
                    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                        eng_sentence, ch_sentence)
                    predictions = transformer(eng_sentence,
                                              ch_sentence,
                                              encoder_self_attention_mask.to(device),
                                              decoder_self_attention_mask.to(device),
                                              decoder_cross_attention_mask.to(device),
                                              enc_start_token=False,
                                              enc_end_token=False,
                                              dec_start_token=True,
                                              dec_end_token=False)
                    next_token_prob_distribution = predictions[0][word_counter]  # not actual probs
                    next_token_index = torch.argmax(next_token_prob_distribution).item()
                    next_token = index_to_chinese[next_token_index]
                    ch_sentence = (ch_sentence[0] + next_token,)
                    if next_token == END_TOKEN:
                        break

                print(f"Evaluation translation (should we go to the mall?) : {ch_sentence}")
                print("-------------------------------------------")

        print("Saving Checkpoint for epoch {}...".format(epoch))
        torch.save({'epoch': epoch,
                    'model_state_dict': transformer.state_dict(),
                    'optimizer_state_dict': optim.state_dict()},
                   # 'checkpoints/model_{}.pth'.format(epoch)
                   'checkpoints/model.pth'
                   )
        artifact = wandb.Artifact('model', type='model', metadata={"epoch":epoch})
        artifact.add_file('checkpoints/model.pth')
        run.log_artifact(artifact)


def translate(eng_sentence):
    eng_sentence = (eng_sentence,)
    ch_sentence = ("",)
    for word_counter in range(max_sequence_length):
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            eng_sentence, ch_sentence)
        predictions = transformer(eng_sentence,
                                  ch_sentence,
                                  encoder_self_attention_mask.to(device),
                                  decoder_self_attention_mask.to(device),
                                  decoder_cross_attention_mask.to(device),
                                  enc_start_token=False,
                                  enc_end_token=False,
                                  dec_start_token=True,
                                  dec_end_token=False)
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = index_to_chinese[next_token_index]
        if next_token == END_TOKEN:
            break
        ch_sentence = (ch_sentence[0] + next_token,)
    return ch_sentence[0]

def evaluate():
    transformer.eval()
    eval_eng_sentences = [
        "what should we do when the day starts?",
        "should we go to the mall?",
        "what is the time?",
        "what is the weather like?",
        "how are you?",
        "what is the meaning of life?",
        "I am going to the store",
        "I will be back soon",
        "My name is Shiqi Ai",
        "I am a student",
        "I am a software engineer",
        "I am a data scientist",
        "I am a machine learning engineer",
        "I am a deep learning engineer",
        "I am a computer vision engineer",
        "I am a natural language processing engineer",
        "I like to play basketball",
        "I like to play soccer",
    ]

    eval_translations = []
    for eng_sentence in eval_eng_sentences:
        eng_sentence = eng_sentence.lower()
        translation = translate(eng_sentence)
        eval_translations.append(translation)

    for eng_sentence, translation in zip(eval_eng_sentences, eval_translations):
        print(eng_sentence)
        print(translation)
        print("-------------------------------------------")


if __name__ == "__main__":
    if args.evaluation_mode:
        evaluate()
    else:
        train()
