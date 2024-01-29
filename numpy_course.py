#~~~~~~~~~NUMPY CRASH COURSE-PATRICK LOEBER~~~~~~~~~~~#

import numpy as np
print(np.__version__)

a = np.array([1,2,3])
print(a)
#bu kod satırı arrayin boyutunu gösterecek çıktısı (3,) şkelindeolacak çünkü 1D array
print(a.shape)
#data type gösterecek int64 olacak bunun için
print(a.dtype)
#number of dimension
print(a.ndim)
#toplam eleman sayını gösterir
print(a.size)
#her elemanın byte türünden boyutunu gösterir bu elemanlar 8 byte
print(a.itemsize)
#elemanı yazdırırı hangi indexi verirsek
print(a[0])

#********** LIST VS NUMPY *********

l = [1,2,3]
#listeye 4 elemanı eklendi
l.append(4)
#append komutunu array de denersek hata alırız

l =[1,2,3]
# a = np.array([1,2,3])
# - l.append(4) : komutuyla listeye 4 elemanını ekleyebiliriz ancak aynı işlemi numpy array için yapamayız çünkü numpy array de append kodu yoktur.

# - l = l + [4] : komutuyla önceki listemize bu yeni liste eklenir. bunu print edersek [1,2,3,4] çıktısını alırız.
#aynı kodu 
a = a + np.array([4]) #şeklinde yapmaya çalışırsak bunda yukarıdaki gibi çıktı almayız. bunun çıktısı [5,6,7] şeklinde olacaktır.
#anlayacağınız üzere 4 rakamı her bir eleman ile topanarak yeni bir liste haline geldi.
a = a + np.array([4,4,4]) #şeklinde yazarsak da aynı çıktıyı alırız ancak buna gerek yoktur tek bir 4 yazdığımızda da numpy bunu anlayacaktır.

l = l * 2 # kodunun çıktısı [1,2,3,1,2,3] şeklinde olacaktır l listesinin 2 kere yan yana yazılmış hali. bunu array de deneyecek olursak;
a = a * 2 # bunun çıktısı [2,4,6] şeklinde elemanların 2 ile çarpılmış hali olacaktır.
a = np.sqrt(a) # bu işlem elemanların karakökünü alır.
a = np.log(a) # bu işlem elemanların logaritmasını alır.

#********** DOT PRODUCT *********

l1 = [1,2,3]
l2 = [4,5,6]

dot = 0
for i in range(len(l1)):
    dot += l1[i] * l2[i]
print(dot)

#numpy da dot operatörü vektörlerin noktasal çarpımı olarak geçer
#İki vektörün boyutları (uzunlukları) aynı olmalıdır. Sonuç, bir skalar (tek bir sayı) olacaktır.
#aynı indexte bulununa elemanların çarpımlarının toplamlarına eşittir.
# = 1*4 + 2*5 + 3*6 şeklinde sonucu bulur
a1 = np.array(l1)
a2 = np.array(l2)

dot = np.dot(a1,a2)
print(dot)

#aynı işlemi şu şekilde de yapabiliriz
sum1 = a1 * a2
dot = np.sum(sum1)
#bu da aynı şekilde çıktı verir
dot = (a1 * a2).sum()
print(dot)

#buradaki @ de matrs çarpımını ifade eder.
dot = a1 @ a2
print(dot)

#********** SPEED TEST **********

from timeit import default_timer as timer

a = np.random.randn(1000)
b = np.random.randn(1000)

A = list(a)
B = list(b)

T = 1000

def dot1():
    dot = 0
    for i in range (len (A)):
        dot += A[i] * B[i]
    return dot

def dot2():
    return np.dot(a,b)

start = timer()
for t in range(T):
    dot1()
end = timer ()
t1 = end - start

start = timer()
for t in range(T):
    dot2()
end = timer ()
t2 = end - start

print('list calculation', t1)
print('np.dot', t2)
print('ratio', t1/t2)
#buradan da gördüğümüz üzere numpy array liste göre çok çok daha hızlıdır.

#********** MULTIDIMENSIONAL(ND) ARRAYS **********

#2 dimensional array
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a.shape) # (2/row,2/column) virgülle ayrılan yerler satır ve sütün gösterir aynı zamanda buradaki 2 satırda 2 eleman olduğunu gösterir.

print(a[0]) #bu kod ilk satırı tamamen yazdırır 

#ilk satırın ilk elemanına ulaşmak istersen ;
print(a[0][0])
print(a[0,0]) #şeklinde de yazabilirsin

#********** INDEXING/SLICING/BOOLEAN INDEXING **********

print(a[:,0]) #ilk sütunu getirir yani 1 ve 3 ([1,2],[3,4])
print (a[0,:])# ilk satırı getirir

print(a.T) #matrixin transpozunu alır

#print(np.linalg.inv(a)) #matrisin tersini alır

#print(np.linalg.det(a)) #matrisin determinantını alır

print(np.diag(a)) # matrixin diogonalini bulur

c = np.diag(a)
print(np.diag(c)) # bu kod bloğuyla diogonal matris bulunur. köşegen dışındaki elemanlar sıfırdır.

b = a[0,:] #ilk satırı yazdırır
print(b)

b = a[0,1:3] #0.satırın 1. indeksiyle 3. indeksi arasını al 3 dahil değil 1 ve 2
print(b)

b = a[:,0] #ilk sütunu yazdırır
print(b)

#boolean indexing
a = np.array([[1,2],[3,4],[5,6]])
print(a)

bool_idx = a > 2 #verilen şartı her bir elemanın doğrulayıp doğrulamadığına göre true false dan oluşan bir matrix yazdırır
print(bool_idx)

print(a[bool_idx]) #sağlayan elemanları 1D olacak şekilde yazdırır
print(a[a > 2]) #bu da yukarıdakiyle aynı işlemi yapar

b = np.where(a>2, a, -1) #ilk kısım koşul her bir eleman içn kontrol edilir doğruysa o elemanın değeri a matrisinde aynen kullanılır yanlışsa ve eşitse elemanın değeri -1 olarak atanır
print(b)

# fancy indexing
a = np.array([10,19,30,41,50,61])
print(a)
b = [1,3,5]
print(a[b])# b nin elemnaları burda indis numarası olarak işlem görür

even = np.argwhere(a%2==0).flatten()
#np.argwhere() koşulu sağlayan elemanların indexlerini bulur.
#flatten() metodu elde edilen indekslerin iç içe geçmiş bir dizi olmasını engeller ve bunları düzleştirir, yani tek boyutlu bir diziyi temsil eden bir NumPy dizisi döndürür
print (a[even])

#********** RESAHPING **********
a = np.arange(1,7)# verilen aralığı içeren bir sayı dizisi oluşturur.
print(a)
print(a.shape)

b = a.reshape((2,3))# 2 row, 3 column olacak şekilde bir 2D matrix oluşturur
print(b)

b = a[np.newaxis, :] #boyut ekleme işlemi yapar bu bir satır ekleme işlemidir.
#1 satır 6 sütun olacak şekilde görünüyor
b = a[:, np.newaxis] #6 satır 1 sütun olacak şekilde elemanları dağıttı

#********** CONCATENATION **********

#concatenation fonksiyonu dizileri birleştirmek için kullanılır.
#belirtilen eksen boyunca bir vey adaha fazla diziyi birleştirir
#np.concatenate((array1,array2, ...), axis=0, out=None)
#axis :birleştirmenin yapılacağı eksenin indeksini belirtir.default olarak 0 dır. satırlar boyunca birleşme
#out : opsiyonel olarak sonucun atanacağı bir dizi belirtir
a = np.array([[1,2],[3,4]])
print(a)
b = np.array([[5,6]])
c = np.concatenate((a,b),axis=None)# tek bir satır şekilde yazar
print(c)

#--hstack--#
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
#verilen dizileri horizontal(yatay) olarak birleştirir önce a sonra b nin elemanları
c = np.hstack((a,b))
print(c)
#verilen dizileri vertical(dikey) olarak birleştirmek için
c = np.vstack((a,b)) #1 satır a vektörü 2. satır vektörü olur
print(c)

#********** BROADCASTING **********

x = np.array([[1,2,3],[4,5,6],[1,2,3],[4,5,6]])
a = np.array([1,0,1])
y = x + a
print(y)
#bu işlem sayesinde x in her bir satırına a vectörü eklenecektir. her satırda aynı sayıda eleman var ekelenecek olan vektörde aynı şekilde.
#a arrayini tek satır halinde yazdığımızda her satıra ekledi eğer her satıra farklı şeyler eklemek istersem bunun için [[1,0,1],[2,1,0],[1,0,1],[2,1,0]] şkelinde bir array yazarak her satır kendi satırıyla toplanacak şekilde işlem yapabiliriz.


#********** FUNCTION AND AXIS **********

a = np.array([[7,8,9,10,11,12,13],[17,18,19,20,21,22,23]])
print(a)
print(a.sum())#elemanların toplamını gösterir
#sum(axis = None) default olarak bu şekildedir istersen sınırlandırma getirebilirsin
print(a.sum(axis=0)) #arraydeki iki satırı toplayıp tek bir x ekseni ahline getirdi
#axis=0 parametresi toplamın sütunlar boyunca yapılmasını belirtir
#axis=1 ise satırları toplamayı ifade eder.
print(a.sum(axis=1))
#bu axis ifadelerini mean fonksiyonu ile de kullanabiliriz. mean ortalama hesaplar
print(a.mean(axis=1))#satırların ortalamasını bulur.
print(a.mean(axis=0))# sütunşların ortalamasını bulur.

# a.var(): Varyans, bir veri setindeki değerlerin ne kadar dağınık olduğunu ölçen bir istatistiksel ölçüdür. Daha spesifik olarak, bir dizi elemanın varyansı, o elemanın ortalamadan ne kadar uzak olduğunu ölçer.
print(a.var(axis = None))
# a.std():NumPy dizisinin standart sapmasını hesaplamak için kullanılır. Standart sapma, bir veri setindeki değerlerin ortalamadan ne kadar uzak olduğunu ölçen bir istatistiksel ölçüdür. Standart sapma, varyansın karekökü alınarak elde edilir.
print(a.std(axis = None))
print(np.std(a, axis = None))# bu da yukarıdakiyle aynı işlemi yapar.
# axis özelliğini min() ve max() fonksiyonlarında da kullanabiliriz.
print(a.max(axis = None))
print(a.min(axis = None))


#********** DATATAYPES **********

x = np.array([1,2])
print(x)
print(x.dtype)
#numpy data typeını otomatik olarak belirler 
#bunu istersek tanımlama yaprken kendimiz belirleyebiliriz
x = np.array([1.2,2.6], dtype=np.int64)
print(x)#data type int olunca değerler de int olarak görünür
print(x.dtype)#değerler float olmasına rağmen biz öyle istediğimiz için data type int 64 olarak görünür

#********** COPYING **********

a = np.array([1,2,3])
b = a# bu işlem sayesinde iki dizide aynı locationı kullanadığı için bir dizide yaptığın diğer diziyi etkileyecek
b[0] = 42
print(b)
print(a)
#bu işlem gerçekleşmesin aynı memory location ı kullanmasınlar diyorsan
a = np.array([1,2,3])
b = a.copy()
#şeklinde bir b dizisi oluştururuz bu durumda bunlar aynı referansı kullanmamış olur.
b[0] = 42
print(b)
print(a)

#********** GENARATING ARRAYS **********

#2 satırlık 3 sütunluk bir 0 matrixi oluşturur.
a = np.zeros((2,3))
print(a)

#2 satırlık 3 sütunluk bir 1 matrixi oluşturur.
a = np.ones((2,3))
print(a)
#default olarak bunların dtype ı floatdır
#aynı işlemi yapmak ama matrixi dolduracağımız değeri kendimiz seçmek istersek 

a = np.full((2,3), 5.0) #2 satırlık 3 sütunluk bir matrix oluşturur ve içini 5 ile doldurur.
print(a)

#ıdentitiy(birim) matrix oluşturmak için ise
a = np.eye((3))# tek argüman alır ve bir kare matris oluşturur
print(a) #köşegeni 1 gerisi 0 olan matris

a = np.arange(20)
#argüman olarak yazılan değere kadar bir 1D arrya oluşturur 0 dahil 20 ye kadar.
print(a)

a = np.linspace(0,10,5)
# belirli bir aralıktaki sayıları belirli bir sayıda eşit aralıklı parçaya bölmek için kullanılır. 
# ilk parametreden başlar bu ilk elemandır 
# 2. parametrede son bulur bu da son elemandır.
# 3. parametre ise kaç elemanlı olacağını gösterir buna göre aradaki değerler belirlenir

#********** RANDOM NUMBERS **********

#bu işlem 0-1 arasında random sayı üretir ve belirtilen matrixi doldurur.
a = np.random.random((3,2))
print(a)
#eğer belirli bir ortalama ve varyans istiyorsanız, numpy.random.randn() fonksiyonunu kullanabilirsiniz. 
#Bu fonksiyon, belirli bir ortalamaya ve varyansa sahip normal (Gaussian) dağılıma uyan rastgele sayılar üretir
a = np.random.randn(1000)
#bunda tuple kullanılamıyor yukarıdaki gibi yazamayız içini
#normal/gaussian 
print(a.mean(), a.var())
#mean =~ 0 , var =~ 1 çıkacaktır

a = np.random.randint(3,10,size=(3,3))
#np.random.randint(start,stop, size= (tuple))
#stop dahil edilmez
#np.random.randint(10,size=(3,3)) şeklinde girilseydi alt sınır 0 kabul edilip üst sınır 10 alınırdı
print(a)

a = np.random.choice(5, size=10) 
#10 elemanlı bir dize oluşturup 0 ile 5 arasından sayılar seçerek bunu doldurur
print(a)

a = np.random.choice([-8,-7,-6], size=10) 
print(a)
#burada ise 10 elemanlı dizeyi [-8,-7,-6] bu dizedeki elemanları rastgele sırayla seçerek doldurmasıyla sonuçlanır


#********** LINEAR ALGEBRA (EIGENVALUES / SOLVING LINEAR SYSTEMS) **********

a = np.array([[1,2],[3,4]])
eigenvalues , eigenvectors = np.linalg.eig(a)
print(eigenvalues)
print(eigenvectors)  #column vector 

#e_vec * e_val = A * e_vec

#Bu kontrol, Av=λv denkleminin doğru olduğunu doğrulamaya yöneliktir.
#Eğer sonuç True ise, bu durum doğrulanmış demektir.
#Bu tür bir kontrol, eigenvector ve eigenvalue hesaplamalarının doğruluğunu test etmek için kullanılabilir.

b = eigenvectors[:,0] * eigenvalues[0]
print(b)
c = a @ eigenvectors[:,0]
print(c)

print(np.allclose(b,c))
#np.allclose(b, c): Bu ifade, iki vektörün (b ve c) tüm elemanlarının yaklaşık olarak eşit olup olmadığını kontrol eder.
#Bu, sayısal hesaplamalarda hassasiyet nedeniyle doğrudan eşitlik kontrolü kullanmaktan kaçınmak için yaygın olarak kullanılan bir yöntemdir.

#--Solving Linear System--#

#Ax=b ---> x=inv(A)b
A = np.array([[1,1],[1.5,4.0]])
b = np.array([2200,5050])

x = np.linalg.inv(A).dot(b)
print(x)
#bu yöntem bulacaktır evet ama inv kullanmak en iyi tercih değildir bu hem yavaştır hemde mükemmmel değildir
#bundan daha iyi bir yöntem şudur;
x = np.linalg.solve(A,b)
print(x)

#********** LOADING CSV FILES **********

#CSV uzantılı dosyalardan programımıza data yüklemek istersek kullanabileceğimiz iki kod var
#1. np.loadtxt('filename', delimiter= "" , dtype=np./float32 gibi/)
#2. np.genfromtxt('filename', delimiter= "" , dtype=np./float32 gibi/)
#delimiter parametresi, dosyadaki verilerin nasıl ayrıldığını belirtir. Yani, dosyanın hangi karakterle (virgül, boşluk, tab, vb.) ayrıldığını ifade eder.
#yükleyeceğimiz data dosyasıyla .py uzantılı program dosamız aynı klasörde olmalı




















