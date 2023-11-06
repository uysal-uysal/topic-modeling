### Topic Modeling & Latent Dirichlet Allocation
Topic Modeling, bir metin belgesinde “topics” adı verilen kelime gruplarını bulmak için kullanılan 'unsupervised' bir yaklaşımdır. Bu konular, sık sık birlikte ortaya çıkan ve genellikle ortak bir temayı paylaşan kelimelerden oluşur. 

Latent Dirichlet Allocation (LDA), her belgenin bir konu koleksiyonu olarak kabul edildiği ve belgedeki her kelimenin konulardan birine karşılık geldiği bir topic modeling örneğidir.

Dolayısıyla, bir belge(text data) verildiğinde LDA, belgeyi temel alarak her konu grubunu, o grubu en iyi açıklayan bir dizi kelimenin olduğu konu gruplarına kümeler.


### Kullanılan Algoritmalar ve Kütüphaneler



|  | README                                                                                                                                                                     |
| ------ |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Latent Dirichlet Allocation (LDA) | LDA modeli, metinlerin içerdiği gizli konuları keşfetmek için kullanılır. Gensim kütüphanesi kullanılarak LDA modeli oluşturulur ve eğitilir.                              |
| Text Processing | Metin işleme işlemleri için Python dilinin temel kütüphaneleri kullanılmıştır. Metinler önce işlenir, ardından stop kelimeleri temizlenir ve kelime dağarcığı oluşturulur. |
| SQLite | LDA sonuçlarını ve Google arama sonuçlarını saklamak için SQLite kullanılmıştır.                                                                                           |
| Web Scraping | Google arama sonuçlarını çekmek ve analiz etmek için BeautifulSoup kullanılmıştır.                                                                                         |
| Tabulate | Tabulate kütüphanesi, sonuçları tablolar halinde görüntülemek için kullanılmıştır.                                                                                         |


## Kurulum
Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyiniz:


```bash
   git clone https://github.com/uysal-uysal/topic-modeling.git
``` 

```bash
  cd topic-modeling
```

```bash
  pip install -r requirements.txt
```

   - [[Gensim](https://radimrehurek.com/gensim/)] [[pandas](https://pandas.pydata.org/)] [[pyshorteners](https://pypi.org/project/pyshorteners/)] [[requests](https://requests.readthedocs.io/en/latest/)] [[beautifulsoup4](https://pypi.org/project/beautifulsoup4/)] [[tabulate](https://pypi.org/project/tabulate/)]
  
    


