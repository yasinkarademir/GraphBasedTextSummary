import tkinter as tk
from tkinter import filedialog
import docx

import tkinter as tk
from tkinter import filedialog
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def create_graph_from_sentences(sentences):
    G = nx.Graph()

    for i in range(len(sentences) - 1):
        G.add_edge(sentences[i], sentences[i + 1])

    return G



# def visualize_graph(G):
#   pos = nx.spring_layout(G)
#  nx.draw(G, pos, with_labels=True, node_size=1000, font_size=10, font_weight='bold')
# plt.show()


def visualize_graph(G, parent_widget):
    fig = Figure(figsize=(100, 100))
    ax = fig.add_subplot(111)

    pos = nx.spring_layout(G, k=1.5)
    nx.draw(G, pos, with_labels=True, node_size=1000, font_size=10, font_weight='bold', ax=ax)

    canvas = FigureCanvasTkAgg(fig, master=parent_widget)
    canvas.draw()
    canvas.get_tk_widget().pack()


def select_file(root):
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = file.read().split(".")
        for sentence in sentences:
            print(sentence)

        G = create_graph_from_sentences(sentences)
        visualize_graph(G, root)


root = tk.Tk()

root.geometry("100x100")
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
