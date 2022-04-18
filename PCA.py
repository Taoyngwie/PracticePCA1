#โดยเวิคช็อปนี้จะมีเพียงเเค่ข้อมูลดิบเเล้วจำทำการกรองเพื่อให้ได้ข้อมูลตามที่เราต้องการ
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
x,y=make_blobs(n_samples=100,n_features=10) #จำลองโดยการสุ่มสร้างข้อมูลมา 100 เเถว 10 คอลัมน์

print(x.shape)
print(y.shape)

#เรียก pca 
print("Before = ",x.shape)
pca=PCA(n_components=4)
x=pca.fit_transform(x)
print("After = ",x.shape)

#เเสดงผลค่าความผันเเปรของเเต่ละ Component
df=pd.DataFrame({'var':pca.explained_variance_ratio_,'pc':['PC1','PC2','PC3','PC4']})
sb.barplot(x='pc',y='var',data=df,color='c') #เเสดงค่า
plt.show()
