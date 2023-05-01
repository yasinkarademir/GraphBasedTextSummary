
Person: 
- GRAF TABANLI METİN ÖZETLEME PROJESİ


Bu projede verilen bir dokümandaki cümlelerin graf yapısına dönüştürülmesi ve bu graf
modelinin görselleştirilmesi istenmektedir. Ardından graf üzerindeki düğümler ile özet
oluşturan bir algoritma oluşturulması beklenmektedir.
Amaç: Proje gerçekleştirimi ile öğrencilerin veri yapıları bilgisinin pekiştirilmesi ve problem
çözme becerisinin gelişimi amaçlamaktadır.
Programlama Dili: Proje C++, C#, Java veya Python dili kullanılarak gerçekleştirilecektir.
Projede aşağıdaki isterleri yerine getirmeniz beklenmektedir.
İSTERLER:
Projede masaüstü uygulama geliştirmeniz gerekmektedir. Masaüstü uygulamada ilk olarak
doküman yükleme işlemi gerçekleştirilecektir. Ardından yüklenen dokümandaki cümleleri
graf yapısı haline getirmeniz ve bu graf yapısını görselleştirmeniz beklenmektedir. Bu grafta
her bir cümle bir düğümü temsil edecektir. Cümleler arasındaki anlamsal ilişki kurulmalı,
cümleler skorlanmalıdır. Belirli parametreleri kullanarak cümle skorunun hesaplama
algoritmasını ve cümle skorlarına göre metin özeti çıkarma algoritmalarını sizin geliştirmeniz
istenmektedir. Özet metni arayüzde sunmanız beklenmektedir. Sonuç olarak size verilen bir
metnin özetini bu yöntem ile çıkarmanız ve gerçek özet ile benzerliğini “ROUGE” skorlaması
ile ölçmeniz istenmektedir.

Projede temel amaç; cümleleri graf yapısına çevirip Cümle Seçerek Özetleme (Extractive
Summarization) gerçekleştirmektir. Graf yapısına çevirerek cümlelerin metindeki anlamsal
ilişkilerini görselleştirmek ve bu ilişkileri kullanarak önemli cümleleri belirlemek
amaçlanmaktadır.
Masaüstü Arayüzü Geliştirilmesi ve Graf Yapısının Oluşturulması
● Masaüstü arayüzü geliştirmeniz beklenmektedir. Arayüz aşağıdaki isterleri
içermelidir:
○ Kullanıcının doküman yükleyebileceği bir alan,
○ Dokümanın graf halinde görüntüleneceği bir alan,
○ Cümle benzerliği için threshold seçilebilecek bir araç,
○ Cümle skorunun belirlenmesi için threshold seçilebilecek bir araç.
○ Cümle benzerliği algoritmasına alternatif oluşturursanız bunun arayüzden
seçilebilmesini sağlayan bir araç.
● Dokümandaki cümleleri graf yapısına dönüştürmek için hazır bazı veritabanları,
kütüphaneler veya API kullanabilirsiniz. Bunlardan bazıları;
○ Neo4j: Grafik veritabanı yönetim sistemi olarak bilinir ve grafik yapısını
kullanarak verileri depolar ve işler.
○ NetworkX: Python programlama dili için açık kaynaklı bir graf
kütüphanesidir. Düğümler ve kenarlar gibi grafik elemanlarını temsil etmek
için birden fazla graf sınıfı sağlar.
○ Graph-tool: C++ programlama dili için açık kaynaklı bir graf kütüphanesidir.
hızlı ve büyük veri kümeleri için daha uygun olabilir.
○ Gephi: Java programlama dili için açık kaynaklı bir grafik analiz aracıdır.
Grafiklerin görselleştirilmesine ve analiz edilmesine yardımcı olan bir dizi
araç sağlar
○ igraph: R, Python ve C/C++ için açık kaynaklı bir graf kütüphanesidir.
Düğümler ve kenarlar gibi grafik elemanlarını temsil etmek için birden fazla
graf sınıfı sağlar. Ayrıca, grafikleri manipüle etmek ve farklı ölçütlere göre
analiz etmek için bir dizi fonksiyon sunar.
Cümleler Arası Anlamsal İlişkinin Kurulması
● Cümlelere NLTK kütüphanesi kullanılarak aşağıdaki ön işleme adımları
uygulanmalıdır:
○ Tokenization: Bir metnin küçük parçalara ayrılmasıdır.
○ Stemming: Kelimelerin kökünün bulunması işlemidir.
○ Stop-word Elimination: Bir metindeki gereksiz sözcükleri çıkarma işlemidir.
Stop word'ler, genellikle yaygın olarak kullanılan, ancak metnin anlamını
belirlemekte önemli bir rol oynamayan kelime ve ifadelerdir.
○ Punctuation: Cümledeki noktalama işaretlerinin kaldırılmasıdır.
NOT: Cümle skoru hesaplama adımında yapılması gereken özel isim içerme ve
nümerik veri içerme adımları bu ön işlemlerden önce gerçekleştirilmelidir.
● İki cümle arasındaki anlamsal ilişkiyi kurmak için aşağıdaki yöntemlerden en az biri
kullanılmalıdır (İki yöntemin de kullanılması durumunda ek puan verilecektir):
○ Word Embedding: Kelime düzeyindeki anlamsal ilişkileri yakalamak için
kullanılan bir makine öğrenimi tekniğidir. Cümleleri temsil etmek için word
embedding kullanıldığında, her kelime; vektörleri ile temsil edilir ve cümle
vektörü, içerdikleri kelime vektörlerinin toplamıdır. Bu şekilde, cümlelerin
anlamsal ilişkileri vektör uzayında ölçülebilir hale gelir.
○ BERT: Özellikle doğal dil işleme (NLP) alanında kullanılan bir derin öğrenme
modelidir. BERT, bir cümleyi tamamen anlamak ve cümleyi oluşturan
kelimelerin birbirleriyle olan ilişkilerini anlamak için kullanılabilir. BERT,
önceden eğitilmiş bir modeldir ve büyük bir metin korpusunda önceden
eğitilir. Bu sayede, dildeki örüntüleri ve anlamsal ilişkileri öğrenir ve
genelleştirir.
● Benzerliği ölçmek için “kosinüs benzerliği” yöntemini uygulamalısınız. Kosinüs
benzerliği, iki vektör arasındaki benzerliği ölçmek için kullanıldığı gibi, iki cümle
arasındaki benzerliği de ölçmek için kullanılabilir.
Cümle Skoru Hesaplama Algoritmasının Geliştirilmesi
● Cümle Skoru Hesaplama sırasında aşağıdaki parametreleri oluşturmalısınız:
○ Cümle özel isim kontrolü (P1)
Cümledeki özel isim sayısı / Cümlenin uzunluğu
○ Cümlede numerik veri olup olmadığının kontrolü (P2)
Cümledeki numerik veri sayısı / Cümlenin uzunluğu
○ Cümle benzerliği threshold’unu geçen node’ların bulunması (P3)
Tresholdu geçen nodeların bağlantı sayısı / Toplam bağlantı sayısı
○ Cümlede başlıktaki kelimelerin olup olmadığının kontrolü (P4)
Cümledeki başlıkta geçen kelime sayısı / Cümlenin uzunluğu
○ Her kelimenin TF-IDF değerinin hesaplanması (P5). Buna göre dokümandaki
toplam kelime sayısının yüzde 10'u ‘tema kelimeler’ olarak belirlenmelidir.
Cümlenin içinde geçen tema kelime sayısı / Cümlenin uzunluğu
NOT: “TF-IDF, bir metinde belirli bir kelimenin ne kadar önemli olduğunu belirlemek için
kullanılan bir istatistiksel yöntemdir. Bu, belirli bir kelimenin ne kadar yaygın olduğunu ve
ne kadar nadir olduğunu dikkate alır.
Aşağıdaki formül kullanılarak TF-IDF değeri hesaplanabilir:
TF-IDF = TF x IDF
Daha fazla bilgi için: https://mdurmuss.github.io/tf-idf-nedir/
Yukarıdaki parametrelerin hepsini kullanarak cümle skorlamak için bir algoritma
geliştirmeniz beklenmektedir. Algoritma sonucunda her bir node un skoru oluşmalıdır.

Skorlara Göre Metin Özetleme Algoritmasının Geliştirilmesi
● Önemli cümleler üzerinden gidilerek özet çıkarılacaktır. Özet çıkarmada kullanılan
bazı yöntemler şunlardır;
○ Cümle seçerek özetleme: Burada amaç metin içerisindeki önemli cümleleri
puanlandırma yöntemleri kullanarak, istatistiksel metotlar ve sezgisel
yaklaşımlar ile cümle seçmektir.
○ Yorumlayarak özetleme : Bu tip özetlemedeki amaç metin içerisindeki
cümlelerin kısaltılmasıdır.
Projede cümle seçerek özetleme yapılmalıdır, yani “var olan cümle yapısı bozulmadan
cümleler seçilerek çıkarılıp özet elde edilecektir”. Oluşan node skorlarına göre node seçip
bunlar ile özet oluşturacak bir metin özetleme algoritması geliştirmeniz beklenmektedir.
Algoritmanızda metin özetlenirken hangi cümlelerin hangi sıra ile seçileceğini, cümle
skorlarını kullanarak sizin belirlemeniz gerekmektedir. Oluşturulan özet arayüzde
gösterilmelidir.
Özetleme Başarısının ROUGE Skoru ile Hesaplanması
● Algoritma sonucu oluşan Özet ile metnin gerçek özeti arasındaki benzerliği ROUGE
skoru ile hesaplamalısınız.”ROUGE” skoru, iki metnin benzerliğini ölçmek için
kullanılır. Bu benzerlik genellikle referans metinde bulunan kelimelerin özetlenmiş
metinde de bulunup bulunmadığına dayanır. Size verilen bir dokümanı özetlemeniz ve
yine size verilecek gerçek özet ile karşılaştırmanız istenmektedir.

ChatGPT: 
- Graf Tabanlı Metin Özetleme Projesi Özeti

Bu projede, verilen bir dokümandaki cümleleri graf yapısına dönüştürerek, cümleler arasındaki anlamsal ilişkileri görselleştirmeyi ve bu ilişkileri kullanarak önemli cümleleri belirleyerek metin özetleme işlemi gerçekleştirmeyi amaçlamaktadır. Aşağıda projenin ana adımları özetlenmiştir: 
1. Masaüstü arayüz geliştirme ve graf yapısının oluşturulması:
- Kullanıcıların doküman yükleyebileceği, graf görselleştirebileceği ve özetleme algoritmasına parametreler sağlayabileceği bir masaüstü arayüz geliştirin.
- Cümleleri graf yapısına dönüştürmek için uygun bir graf kütüphanesi veya API kullanın (Neo4j, NetworkX, Graph-tool, Gephi, igraph vb.). 
2. Cümleler arası anlamsal ilişkinin kurulması:
- Cümlelere NLTK kütüphanesi kullanarak ön işleme adımları uygulayın (Tokenization, Stemming, Stop-word Elimination, Punctuation).
- İki cümle arasındaki anlamsal ilişkiyi kurmak için Word Embedding veya BERT gibi yöntemler kullanın.
- Kosinüs benzerliği ile cümleler arasındaki benzerliği ölçün. 
3. Cümle skoru hesaplama algoritmasının geliştirilmesi:
- Cümle skorlaması için belirtilen parametreleri kullanarak (P1, P2, P3, P4, P5) bir algoritma geliştirin.
- Algoritma sonucunda her bir düğüm için bir skor elde edin. 
4. Skorlara göre metin özetleme algoritmasının geliştirilmesi:
- Cümle seçerek özetleme yöntemi kullanarak, cümle skorlarını kullanarak özet oluşturun.
- Oluşturulan özetin arayüzde gösterilmesini sağlayın. 
5. Özetleme başarısının ROUGE skoru ile hesaplanması:
- Algoritma sonucu oluşan özet ile metnin gerçek özeti arasındaki benzerliği ROUGE skoru ile hesaplayın.

Projenin başarılı bir şekilde tamamlanması, öğrencilerin veri yapıları bilgisini pekiştirmelerine ve problem çözme becerilerinin gelişmesine katkıda bulunacaktır. Projeyi C++, C#, Java veya Python dili kullanarak gerçekleştirebilirsiniz.

Person: 
- python ile geliştireceğim. nasıl yapacağımı anlat

ChatGPT: 
- Python ile bu projeyi gerçekleştirmek için aşağıdaki adımları takip edebilirsiniz: 
1. Gerekli Kütüphanelerin İçe Aktarılması:
- NLTK, numpy, sklearn, networkx, rouge gibi kütüphaneleri kullanarak başlayın.

```python

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from rouge import Rouge
```

 
1. Metin Ön İşleme:
- NLTK kütüphanesi kullanarak tokenization, stemming, stop-word elimination ve punctuation işlemlerini gerçekleştirin.

```python

def preprocess_text(text):
    # tokenization, stemming, stop-word elimination ve punctuation işlemleri
    return preprocessed_text
```

 
1. Cümle Benzerliği Hesaplama:
- Word Embedding veya BERT kullanarak cümle benzerliği hesaplaması yapın.

```python

def calculate_sentence_similarity(sentence1, sentence2):
    # cümle benzerliği hesaplaması
    return similarity_score
```

 
1. Graf Oluşturma ve Görselleştirme:
- NetworkX kütüphanesini kullanarak graf oluşturun ve düğümleri cümlelerle doldurun.

```python

def create_sentence_graph(sentences):
    # graf oluşturma ve düğümleri doldurma
    return graph
```

 
1. Cümle Skoru Hesaplama Algoritması:
- Cümle skoru hesaplama algoritması geliştirin ve cümle skorlarını kullanarak her düğümün skorunu hesaplayın.

```python

def calculate_sentence_scores(graph, sentences):
    # cümle skoru hesaplama algoritması
    return sentence_scores
```

 
1. Metin Özetleme Algoritması:
- Cümle skorlarını kullanarak özet metni oluşturun.

```python

def extractive_summarization(sentence_scores, sentences):
    # metin özetleme algoritması
    return summary
```

 
1. ROUGE Skoru Hesaplama:
- Oluşan özet ile gerçek özet arasındaki ROUGE skorunu hesaplayın.

```python

def calculate_rouge_score(summary, reference_summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference_summary)
    return scores
```

 
1. Ana Fonksiyon:
- Yukarıdaki tüm fonksiyonları bir araya getirerek projeyi çalıştırın.

```python

def main():
    # metin yükleme, ön işleme, cümle benzerliği hesaplama, graf oluşturma,
    # cümle skoru hesaplama, metin özetleme ve ROUGE skoru hesaplama adımlarını gerçekleştirin.

if __name__ == "__main__":
    main()
```



Bu adımlar temel olarak projenin ana hatlarını oluşturur. Her adımı özelleştirerek, projeniz için daha spesifik gereksinimlere uygun hale getirebilirsiniz.
