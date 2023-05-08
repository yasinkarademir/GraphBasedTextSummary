import tkinter as tk
from tkinter import filedialog
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer

import mplcursors

import re

import warnings


# warnings.filterwarnings("ignore")


def create_graph_from_sentences(sentences):
    G = nx.Graph()

    for i in range(len(sentences) - 1):
        G.add_edge(i, i + 1)

    return G


def ozel_isim_kontrolu(cumle):
    ozel_isimler = re.findall(r'\b[A-Z][a-z]*\b', cumle)
    return len(ozel_isimler) / len(cumle.split())


def numerik_veri_kontrolu(cumle):
    numerik_veriler = re.findall(r'\b\d+\b', cumle)
    return len(numerik_veriler) / len(cumle.split())


def baslik_kelime_kontrolu(cumle, baslik):
    baslik_kelimeler = baslik.split()
    cumle_kelimeler = cumle.split()
    ortak_kelime_sayisi = len(set(baslik_kelimeler) & set(cumle_kelimeler))

    return ortak_kelime_sayisi / len(cumle_kelimeler)


def dokumandaki_tema_kelimeleri(metin):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(metin)
    feature_names = vectorizer.get_feature_names()

    # Dokümandaki toplam kelime sayısının yüzde 10'u kadar kelime tema kelimeleri olarak seçilir
    tema_kelimeler = sorted(feature_names, key=lambda x: tfidf_matrix[0, feature_names.index(x)], reverse=True)[
                     :int(len(feature_names) * 0.1)]

    return tema_kelimeler


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
        print(expected_output)

        sentences_input = expected_input.split(".")
        sentences_output = expected_output.split(".")

        sentences_dict = {}
        sentences_dict2 = {}

        # başlık sentences_input[0]
        for idx, sentence in enumerate(sentences_input[1:]):
            if (sentence.strip() != ""):
                # ID, cümle ve cümle puanını (varsayılan 5) sentences_dict2'ye ekle
                sentences_dict2[idx] = {"sentence": sentence.strip() + ".", "score": 5}

        for idx, sentence in enumerate(sentences_output):
            if (sentence.strip() != ""):
                sentences_dict[idx] = sentence.strip()

    # sentences_dict2'yi kullanarak grafiği oluştur
    G = create_graph_from_sentences(sentences_dict2)
    visualize_graph(G, root, sentences_dict2)


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

'''
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Word Dosyası Görüntüleyici")
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.select_file_button = tk.Button(self, text="Dosya Seç", command=self.load_file)
        self.select_file_button.pack(side="top")

        self.file_contents_text = tk.Text(self)
        self.file_contents_text.pack()

    def load_file(self):
        file_path = filedialog.askopenfilename(defaultextension=".docx", filetypes=[("Word Dosyası", "*.docx")])
        if file_path:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                self.file_contents_text.insert("end", paragraph.text + "\n")

root = tk.Tk()
app = Application(master=root)
app.mainloop()
'''
