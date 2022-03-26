#!/usr/bin/env python
# coding: utf-8

# # Konteks

# Bob telah memulai perusahaan selulernya sendiri. Dia ingin memberikan perlawanan keras kepada perusahaan besar seperti Apple, Samsung, dll.
# 
# Dia tidak tahu bagaimana memperkirakan harga ponsel yang dibuat oleh perusahaannya. Di pasar ponsel yang kompetitif ini, Anda tidak bisa begitu saja berasumsi. Untuk mengatasi masalah ini ia mengumpulkan data penjualan ponsel dari berbagai perusahaan.
# 
# Bob ingin mengetahui beberapa hubungan antara fitur ponsel (misalnya: - RAM, Memori Internal dll) dan harga jualnya. Tapi dia tidak begitu pandai Machine Learning. Jadi dia membutuhkan bantuan Anda untuk menyelesaikan masalah ini.
# 
# Dalam masalah ini Anda tidak perlu memprediksi harga sebenarnya tetapi kisaran harga yang menunjukkan seberapa tinggi harganya.
# 
# 0 adalah terendah dan 4 adalah yang tertinggi.

# # Import Library dan Dataset

# Pertama kita import pandas terlebih dahulu untuk membaca file csv.

# In[1]:


import pandas as pd


# Membaca file mobile.csv yang berada pada folder dataset.

# ## Import Dataset

# In[2]:


mobile = pd.read_csv('dataset/mobile.csv')


# Kemudian kita bisa melihat dataset mobile kita dengan menulis variable mobile.

# In[3]:


mobile


# ## Info Dataset

# Kita juga harus mengecek info dari dataset kita dengan fungsi `info()`, fungsi berikut memberi tau semua data kita dari tipe data, nilai null, kolom baris, dan nama atribut.

# In[4]:


mobile.info()


# ## Describe Dataset

# Kita juga akan melihat deskripsi dari dataset dengan menggunakan fungsi `describe()`

# In[5]:


mobile.describe()


# ## Isnull Fungtion

# In[6]:


mobile.isnull().sum()


# # Univariate Analysis

# Pada tahap ini kita akan cek Seluruh data kita dengan library seaborn.

# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[8]:


mobile.hist(bins=50, figsize=(20,15))
plt.show()


# In[9]:


mobile['price_range'].value_counts()


# Dari data di atas kita bisa mengetahui bahwa jumlah data label setara dengan jumlah masing-masing adalah 500 sample

# # Multivariate Analysis

# In[10]:


sns.pairplot(mobile, diag_kind='kde')


# # Correlation Matrix

# Setelah itu kita akan mencoba mencari Korelasi antar variable dengan fungsi `heatmap()` dengan fungsi tersebut kita bisa melihat korelasi pada data kita.

# In[11]:


plt.figure(figsize=(10, 8))
correlation_matrix = mobile.corr().round(6)
 
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix", size=20)


# In[12]:


mobile.drop(['touch_screen'], inplace=True, axis=1)
mobile.head()


# In[13]:


mobile.drop(['mobile_wt'], inplace=True, axis=1)
mobile.info()


# Pada dua code cell di atas kita menghapus beberapa variable dari dataset kita yang memiliki korelasi paling kecil di antara yang lain. Karena kita ingin memaksimalkan proses latihan kita dengan data yang powerfull, data yang memiliki sangat sedikit korelasi bisa mempengaruhi proses latih.

# # Split data ke Training dan Test

# Kita membagi data kita menjadi data latih dan data uji, dengan pembagian 20% untuk data uji.

# In[14]:


from sklearn.model_selection import train_test_split

X = mobile.drop(['price_range'], axis=1)
y = mobile['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)


# Kemudian kita akan mencoba melihat pembagian data kita, dengan mengetikan code di bawah.

# In[15]:


print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')


# Dengan hasil di atas kita bisa menyimpulkan, bahwa,
# * Data training menjadi 1600 sample
# * Data test menjadi 400 sample

# # Standarisasi

# Kita juga akan melakukan Standarisasi agar data kita memiliki rentang yang sama antara 0 dan 1. Ini juga akan membantu membantu pada saat proses pelatihan nanti.

# In[16]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Modeling

# Pada tahap modeling kita akan mencoba 2 algoritma yaitu Random Forest dan juga SVM, kita akan mencari yang terbaik di antara kedua algoritma tersebut.

# ## Random Forest

# Sebelum itu kita akan mencari hyperparameter terbaik untuk dataset kita, kita menggunakan GridSearch untuk mencari parameter terbaik berdasarkan dataset kita.

# In[17]:


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=32)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[18]:


from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)


# In[19]:


CV_rfc.best_params_


# Setelah melakukan GridSeach kita mendapatkan hasil,
# * criterion: entropy
# * max_depth: 8
# * max_features: auto
# * n_estimators: 500
# 
# Parameter itulah yang akan kita pakai pada algoritma RandomForest kita

# In[20]:


RF = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='auto', n_estimators=500)
RF.fit(X_train, y_train)


# In[21]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = RF.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# Dengan RandomForest kita mendapatkan akurasi sekitar 87% cukup baik, namun kita akan menggunakan K-Fold Validasi untuk mencari akurasi yang lebih akurat.

# ## SVM Classifier

# Sama seperti RandomForest kita juga akan mencari parameter terbaik berdasarkan dataset kita.

# In[22]:


from sklearn.svm import SVC
svm = SVC(random_state=32)

param_grid_svm = { 
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
}

CV_rfc = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv= 5)
CV_rfc.fit(X_train, y_train)


# In[23]:


CV_rfc.best_params_


# In[24]:


classifier = SVC(kernel = 'linear', gamma='scale', random_state = 32)
classifier.fit(X_train, y_train)


# In[25]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# Dan hasil yang kita dapatkan luar biasa! Akurasi score mendapat 95%, namun sekali lagi kita akan melihatnya pada K-Fold Validasi untuk melihat lebih detail.

# # k-Fold Cross Validation

# ### RandomForest

# In[26]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = RF, X = X_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ### SVM Classifier

# In[31]:


accuracies_SVM = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies_SVM.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies_SVM.std()*100))


# # Kesimpulan

# Setelah tahap-tahap kita lalui mulai dari melihat dataset, analisis korelasi antar variable, menghilangkan variable yang tidak di perlukan, sampai tahap modeling dan melihat hasilnya bersama.
# 
# Pada tahap `MODELING` kita menggunakan dua jenis algoritma yaitu,
# * RandomForestClassifier
# * SVM Classifier (SVC)
# 
# Dari kedua algoritma tersebut kita bisa tau mana algoritma yang menghasilkan akurasi terbaik, bahkan setelah memnggunakan k-Fold Cross Validation memberikan hasilnya. Algoritma SVM memberikan hasil paling tinggi yaitu 93%, dengan begitu algoritma SVM lah yang akan kita gunakan.

# Akhirnya kita bisa memberi tau kepada Bob kisaran harga ponsel berdasarkan fitur-fitur pada data.

# In[ ]:




