import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('../../experiments/results/results.csv')
df['View']=df['View'].str.replace('.jpg','')
df['View']=df['View'].replace({
    'front':'Front',
    'left45':'L45',
    'left90':'L90',
    'right45':'R45',
    'right90':'R90'
})
pivot=df.pivot_table(
    values='Detection Quality',
    index='View',
    columns='Method',
    aggfunc='mean'
)
pivot.reindex(['Front','L45','L90','R45','R90'])
plt.figure(figsize=(12,6))
views=pivot.index
methos=pivot.columns

n=np.arange(len(views))
width=0.12
for i,method in enumerate(methos):
    plt.bar(
        n+i*width,
        pivot[method],
        width=width,
        label=method
    )
plt.xticks(n+width*len(methos)/2,views)
plt.ylabel('Detection Quality')
plt.xlabel('View')
plt.title('Comparison of Keypoint Detector Across Views')
plt.legend(bbox_to_anchor=(1.05,1),
           loc='upper left',
           borderaxespad=0)
plt.grid(axis='y',linestyle='--',alpha=0.4)
plt.tight_layout()
plt.savefig('../../experiments/results/bar_methods_by_view.jpg',dpi=300)
plt.show()
print(df.head())