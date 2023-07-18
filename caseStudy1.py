###############################################
# Python Alıştırmalar
###############################################

###############################################
# GÖREV 1: Veri yapılarının tipleriniz inceleyiniz.
# Type() metodunu kullanınız.
###############################################

x = 8
type(x)

y = 3.2
type(y)

z = 8j + 18
type(z)

a = "Hello World"
type(a)

b = True
type(b)

c = 23 < 22
type(c)

l = [1, 2, 3, 4, "String", 3.2, False]
type(l)
# Sıralıdır
# Kapsayıcıdır
# Değiştirilebilir


d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
type(d)
# Değiştirilebilir
# Kapsayıcı
# Sırasız
# Key değerleri farklı olacak


t = ("Machine Learning", "Data Science")
type(t)
# Değiştirilemez
# Kapsayıcı
# Sıralı


s = {"Python", "Machine Learning", "Data Science"}
type(s)
# Değiştirilebilir
# Sırasız + Eşsiz
# Kapsayıcı


###############################################
# GÖREV 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
# String metodlarını kullanınız.
###############################################

text = "The goal is to turn data into information, and information into insight."
new_text_2 = text.upper().replace(",", " ").replace(".", " ").split()
new_text_2

###############################################
# GÖREV 3: Verilen liste için aşağıdaki görevleri yapınız.
###############################################

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# step 1

len(lst)

# step 2

lst[0]
lst[10]

# step 3

lst.pop(8)

# step4

lst[0:4]

# step 5

lst.append("X")

# step 6

lst.insert(8, "N")

###############################################
# GÖREV 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
###############################################

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# step 1

dict.keys()

# step 2

dict.values()

# step 3

dict.update({"Daisy": ["England", 13]})
dict.get("Daisy")

# step 4

dict.update({"Ahmet": ["Turkey", 24]})

# step 5

dict.pop("Antonio")
dict

###############################################
# GÖREV 5: Arguman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atıyan ve bu listeleri return eden fonskiyon yazınız.
# Liste elemanlarına tek tek erişmeniz gerekmektedir.
# Her bir elemanın çift veya tek olma durumunu kontrol etmekiçin % yapısını kullanabilirsiniz.
###############################################

l = [2, 13, 18, 93, 22]


def func(numbers):
    odd_numbers = [i for i in numbers if i % 2 == 0]
    even_numbers = [j for j in numbers if j % 2 != 0]

    return odd_numbers, even_numbers


odd, even = func(l)
odd
even

###############################################
# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
###############################################

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

muhendislik = [f"Muhendislik Fakultesi {index} . ogrenci : {ogrenci} " for index, ogrenci in
               enumerate(ogrenciler[:3], 1)]
tip = [f"Tip Fakultesi {index} . ogrenci : {ogrenci} " for index, ogrenci in enumerate(ogrenciler[3:], 1)]

muhendislik
tip
muhendislik + tip

###############################################
# GÖREV 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır.
# Zip kullanarak ders bilgilerini bastırınız.
###############################################


ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

ziped = [f"Kredisi {i} olan {x} kodlu dersin kontenjani {y} kisidir." for i, x, y in zip(kredi, ders_kodu, kontenjan)]
ziped

###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
# Kapsayıp kapsamadığını kontrol etmek için issuperset() metodunu,farklı ve ortak elemanlar için ise intersection ve difference metodlarını kullanınız.
###############################################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])


def check():
    if kume1.issuperset(kume2):
        print(f"2. Kumenin 1. Kumeyle ortak elemanlari : {kume2.intersection(kume1)}")
    else:
        print(f"2. Kumenin 1. Kumeden farki : {kume2.difference(kume1)}")


check()
