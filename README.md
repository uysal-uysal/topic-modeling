### Kullanılan Algoritmalar ve Kütüphaneler

Proje, aşağıdaki algoritmaları ve kütüphaneleri içerir:

| Plugin | README |
| ------ | ------ |
| Latent Dirichlet Allocation (LDA) | LDA modeli, metinlerin içerdiği gizli konuları keşfetmek için kullanılır. Gensim kütüphanesi kullanılarak LDA modeli oluşturulur ve eğitilir. |
| Text Processing | Metin işleme işlemleri için Python dilinin temel kütüphaneleri kullanılmıştır. Metinler önce işlenir, ardından stop kelimeleri temizlenir ve kelime dağarcığı oluşturulur. |
| SQLite | SQLite veritabanı, LDA sonuçlarını ve Google arama sonuçlarını saklamak için kullanılır. SQLite veritabanı kullanılarak yapılan sorgular ve sonuçlar kaydedilir. |
| Web Scraping | BeautifulSoup kütüphanesi, Google arama sonuçlarını çekmek ve analiz etmek için kullanılır. Program, metinlerin konularını belirlemek için Google'da ilgili aramalar yapar. |
| Tabulate | Tabulate kütüphanesi, sonuçları tablolar halinde görüntülemek için kullanılır. Sonuçlar düzenli bir şekilde görüntülenir. |


Proje dosyalarını çalıştırmadan önce gerekli kütüphaneleri aşağıdaki kod parcası ile yüklemiş olmalısınız: 
```bash
pip install -r requirements.txt
```

   - [[Gensim](https://radimrehurek.com/gensim/)] [[pandas](https://pandas.pydata.org/)] [[pyshorteners](https://pypi.org/project/pyshorteners/)] [[requests](https://requests.readthedocs.io/en/latest/)] [[beautifulsoup4](https://pypi.org/project/beautifulsoup4/)] [[tabulate](https://pypi.org/project/tabulate/)]
  
    


