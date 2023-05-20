import tkinter as tk
from tkinter import filedialog
import networkx as nx
from bokeh.palettes import Spectral4
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import mplcursors

import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict, Counter
import string
import ssl
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import nltk
import torch
import numpy as np
import gensim.downloader as api

# SSL sertifika hatası düzeltmesi
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger')

import warnings

# warnings.filterwarnings("ignore")


# Word2Vec modelini yükle
word2vec_model = api.load("word2vec-google-news-300")

# BERT modelini yükle
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def word2vec_sentence_vector(sentence, model):
    words = nltk.word_tokenize(sentence)
    word_vectors = [model[word] for word in words if word in model.key_to_index]
    return np.mean(word_vectors, axis=0)


def bert_sentence_vector(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(1).detach().numpy()


def sentence_similarity(sentence1, sentence2, model_word2vec, model_bert, tokenizer_bert):
    vector1_w2v = word2vec_sentence_vector(sentence1, model_word2vec)
    vector2_w2v = word2vec_sentence_vector(sentence2, model_word2vec)

    vector1_bert = bert_sentence_vector(sentence1, model_bert, tokenizer_bert)
    vector2_bert = bert_sentence_vector(sentence2, model_bert, tokenizer_bert)

    sim_w2v = cosine_similarity([vector1_w2v], [vector2_w2v])[0][0]
    sim_bert = cosine_similarity(vector1_bert, vector2_bert)[0][0]

    return (sim_w2v + sim_bert) / 2


'''
def sentence_similarity(sentence1, sentence2, model_bert, tokenizer_bert):
    vector1_bert = bert_sentence_vector(sentence1, model_bert, tokenizer_bert)
    vector2_bert = bert_sentence_vector(sentence2, model_bert, tokenizer_bert)

    sim_bert = cosine_similarity([vector1_bert], [vector2_bert])[0][0]
    return sim_bert
'''


def sentence_similarity_dict(sentences_dict, model_word2vec, model_bert, tokenizer_bert):
    similarities = {}
    for i in sentences_dict:
        similarities[i] = {}
        for j in sentences_dict:
            if i != j:
                sim = sentence_similarity(sentences_dict[i]["sentence"], sentences_dict[j]["sentence"], model_word2vec,
                                          model_bert, tokenizer_bert)
                similarities[i][j] = sim
    return similarities


def create_graph_from_sentences(sentences):
    G = nx.Graph()

    for i in range(len(sentences) - 1):
        G.add_edge(i, i + 1)

    return G


def cumle_puani_hesapla(cumle, baslik, tema_kelimeler):
    weights = {
        "P1": 0.3,  # özel isim kontrolü
        "P2": 0.3,  # numerik veri kontrolü
        "P3": 0.1,  # başlık kelime kontrolü
        "P4": 0.2  # TF-IDF / tema kelime kontrolü
    }

    p1 = ozel_isim_kontrolu(cumle)
    p2 = numerik_veri_kontrolu(cumle)
    p3 = baslik_kelime_kontrolu(cumle, baslik)
    p4 = tema_kelime_kontrolu(cumle, tema_kelimeler)

    toplam_puan = (p1 * weights["P1"]) + (p2 * weights["P2"]) + (p3 * weights["P3"]) + (p4 * weights["P4"])

    return toplam_puan


def ozel_isim_kontrolu(cumle):
    cumle_kelimeleri = nltk.word_tokenize(cumle)
    cumle_pos = nltk.pos_tag(cumle_kelimeleri)
    ozel_isimler = [kelime for kelime, pos in cumle_pos if pos == 'NNP']
    return len(ozel_isimler) / len(cumle_kelimeleri)


def numerik_veri_kontrolu(cumle):
    numerik_veriler = re.findall(r'\b\d+(?:\.\d+)?(?:st|nd|rd|th|s)?\b', cumle)
    print(numerik_veriler)
    return len(numerik_veriler) / len(cumle.split())


def baslik_kelime_kontrolu(cumle, baslik):
    baslik_kelimeler = baslik.split()
    cumle_kelimeler = cumle.split()
    ortak_kelime_sayisi = len(set(baslik_kelimeler) & set(cumle_kelimeler))

    return ortak_kelime_sayisi / len(cumle_kelimeler)


def on_isleme(cumle):
    stop_words = set(stopwords.words("english"))

    # Cümle ve kelime tokenleştirmesi
    words = word_tokenize(cumle)

    # Noktalama işaretlerini ve durdurma kelimelerini kaldır
    filtered_words = [word.lower() for word in words if word not in stop_words and word not in string.punctuation]

    # Stemming
    # ps = nltk.PorterStemmer()
    # stemmed_words = [ps.stem(word) for word in filtered_words]

    cumle = " ".join(filtered_words)

    return cumle


def tema_kelimeler(text):
    stop_words = set(stopwords.words("english"))

    # Cümle ve kelime tokenleştirmesi
    words = word_tokenize(text)

    # Noktalama işaretlerini ve durdurma kelimelerini kaldır
    filtered_words = [word.lower() for word in words if word not in stop_words and word not in string.punctuation]

    # Kelime frekanslarını say
    word_freq = Counter(filtered_words)

    # Yüzde 10 tema kelime sayısı
    num_theme_words = int(len(filtered_words) * (10 / 100))

    # En yaygın temaları al
    common_theme_words = word_freq.most_common(num_theme_words)

    # Tema kelimelerini döndür
    return [word[0] for word in common_theme_words]


def tema_kelime_kontrolu(cumle, tema_kelimeler):
    cumle_kelimeler = cumle.split()
    tema_kelime_sayisi = len(set(tema_kelimeler) & set(cumle_kelimeler))
    return tema_kelime_sayisi / len(cumle_kelimeler)


def visualize_graph(G, parent_widget, sentences):
    fig = Figure(figsize=(100, 100))
    ax = fig.add_subplot(111)

    pos = nx.spring_layout(G, k=0.3)

    node_labels = {k: f'{k}: {v["score"]}' for k, v in sentences.items()}
    nx.draw(G, pos, node_size=1000, ax=ax, labels=node_labels, font_size=10, font_weight='bold')

    canvas = FigureCanvasTkAgg(fig, master=parent_widget)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Tıklama olayını işleyen mplcursors kullanarak cümleleri ve puanları göster
    def show_sentence(sel):
        index = sel.index  # Düğümün indeksini elde edin (Selection.index kullanılarak)
        sentence = sentences[int(index)]["sentence"]  # İndeksi kullanarak cümleyi elde edin
        score = sentences[int(index)]["score"]  # İndeksi kullanarak puanı elde edin
        sel.annotation.set_text(f"{sentence}\nPuan: {score}")  # Düğümde cümleyi ve puanı göster

    mplcursors.cursor(ax, hover=False).connect("add", show_sentence)


def select_file(root):
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

        expected_input = text.split("(Beklenen Girdi):")[1].split("(Beklenen Çıktı):")[0].strip()
        # print(expected_input.replace(".", "").replace("\n", " ").split(" ")) #inputtaki tüm kelimeler

        expected_output = text.split("(Beklenen Çıktı):")[1].strip()

        # sentences_input = expected_input.split(".")
        sentences_input = expected_input.replace("\n", "").split(".")
        # print(tema_kelimeler(expected_input))

        sentences_output = expected_output.split(".")

        sentences_dict = {}
        sentences_dict2 = {}

        for i in sentences_input:
            print(on_isleme(i))

        # başlık sentences_input[0]
        for idx, sentence in enumerate(sentences_input[1:]):
            if (sentence.strip() != ""):
                # ID, cümle ve cümle puanını (varsayılan 5) sentences_dict2'ye ekle
                sentences_dict2[idx] = {"sentence": sentence.strip() + ".",
                                        "score": cumle_puani_hesapla(sentence.strip(), sentences_input[0],
                                                                     tema_kelimeler(expected_input))}

        for idx, sentence in enumerate(sentences_output):
            if (sentence.strip() != ""):
                sentences_dict[idx] = sentence.strip()
    # -------------------------------------*deneme*-------------------------------------
    '''
    similarity_dict = {}

    for i in range(len(sentences_dict2)):
        for j in range(i + 1, len(sentences_dict2)):
            similarity = sentence_similarity(sentences_dict2[i]['sentence'], sentences_dict2[j]['sentence'],
                                             word2vec_model, bert_model, bert_tokenizer)
            similarity_dict[(i, j)] = similarity

    for pair, similarity in similarity_dict.items():
        print(f"Cümle {pair[0]} ve Cümle {pair[1]} arasındaki benzerlik: {similarity}")
    print(similarity_dict)
    '''

    # -------------------------------------*deneme*-------------------------------------

    # benzerlik sözlüğünü oluştur
    similarity_dict = {}
    for i in range(len(sentences_dict2)):
        for j in range(i + 1, len(sentences_dict2)):
            similarity = sentence_similarity(sentences_dict2[i]['sentence'], sentences_dict2[j]['sentence'],
                                             word2vec_model,
                                             bert_model, bert_tokenizer)
            similarity_dict[(i, j)] = similarity

    # belirli bir eşik üzerindeki benzerliklere sahip cümleleri içeren bir graf oluştur
    similarity_threshold = 0.6
    print(similarity_dict)
    G = create_similarity_graph(similarity_dict, similarity_threshold)

    # grafiği görselleştir
    visualize_similarity_graph(G)

    # -------------------------------------*deneme*-------------------------------------



  # sentences_dict2'yi kullanarak grafiği oluştur
    G = create_graph_from_sentences(sentences_dict2)
    visualize_graph(G, root, sentences_dict2)


def create_similarity_graph(similarity_dict, threshold):
    G = nx.Graph()

    for pair, similarity in similarity_dict.items():
        if similarity > threshold:
            G.add_edge(pair[0], pair[1], weight=similarity)

    return G


def visualize_similarity_graph(G):
    new_window = tk.Toplevel(root)
    new_window.geometry("500x500")
    new_window.title("Benzerlik Grafiği")
    new_window.config(background="#000000")
    new_window.iconbitmap("icon.ico")
    new_window.wm_attributes("-alpha", 0.9)

    fig = Figure(figsize=(30, 30))

    ax = fig.add_subplot(111)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, ax=ax)

    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


root = tk.Tk()

root.geometry("500x500")
# root.resizable(False, False)

root.title("Dosya Yükleme Uygulaması")
root.config(background="#000000")
root.iconbitmap("icon.ico")
root.wm_attributes("-alpha", 0.9)

button = tk.Button(root, text="Dosya Yükle", command=lambda: select_file(root))

button.pack()

root.mainloop()
