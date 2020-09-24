from numpy.random import randn
import numpy as np
np.random.seed(123)
import os
import matplotlib.pyplot as plt
import pandas as pd
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4)
pd.options.display.max_rows = 20

import json
path = 'datasets/bitly_usagov/example.txt'
records = [json.loads(line) for line in open(path)]
# print(records)

time_zones = [rec['tz'] for rec in records if 'tz' in rec]
# print(time_zones[:10])

def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

counts = get_counts(time_zones)
# print(counts)
# print('{} {} {}'.format('America/New_York :', counts['America/New_York'],'times'))
# print(len(time_zones))


# 最常出現時區排名
def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]
# print(top_counts(counts))

# 使用函式庫 collection 類別
# 最常出現時區排名
from collections import Counter
counts = Counter(time_zones)
# print(counts)
# print(counts.most_common()[:10])#時區出現頻率

# 建立Dataframe
import pandas as pd
frame = pd.DataFrame(records)
# print('finish')

# 觀察 Dataframe 資訊
# print(frame.info())

# 觀察 'tz' 前10筆欄位資料
# print(frame['tz'][:10])

# 觀察 'nk' 欄位
# print(frame['nk'][:])

# 有大型資料，可以使用 summary view 預覽資料
tz_counts = frame['tz'].value_counts()
# print(tz_counts[:10])

# 用 fillna 填補 "遺失值" 資料
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'  # 布林索引 換掉字串
tz_counts = clean_tz.value_counts()
# print(tz_counts[:10])

'''''
視覺化
'''''

import seaborn as sns
plt.figure(figsize=(10, 4))
subset = tz_counts[:10]
sns.barplot(y=subset.index, x=subset.values)
# plt.show()


# 使用者使用的瀏覽器資訊
# print(frame['a'][1])
# print(frame['a'][50])
# print(frame['a'][51][:50])  # long line


results = pd.Series([x.split()[0] for x in frame.a.dropna()])
# print(results[:5])                 # 前 5 筆使用者瀏覽器
# print(results.value_counts()[:8])  # 瀏覽器次數計算



cframe = frame[frame.a.notnull()]    # a欄位 非空值資訊
# print(cframe)

cframe = cframe.copy()
cframe['os'] = np.where(cframe['a'].str.contains('Windows'),'Windows', 'Not Windows')
# print(cframe['os'][:5])

by_tz_os = cframe.groupby(['tz', 'os'])
agg_counts = by_tz_os.size().unstack().fillna(0)
# print(agg_counts[:10])


# Use to sort in ascending order
indexer = agg_counts.sum(1).argsort()
# print(indexer[:10])

count_subset = agg_counts.take(indexer[-10:])
# print(count_subset)

# print(agg_counts.sum(1).nlargest(10))


plt.figure()
# Rearrange the data for plotting
count_subset = count_subset.stack()
count_subset.name = 'total'
count_subset = count_subset.reset_index()
count_subset[:10]
sns.barplot(x='total', y='tz', hue='os',  data=count_subset)
# plt.show()


def norm_total(group):
    group['normed_total'] = group.total / group.total.sum()
    return group

results = count_subset.groupby('tz').apply(norm_total)
plt.figure()
sns.barplot(x='normed_total', y='tz', hue='os',  data=results)
# plt.show()

# g = count_subset.groupby('tz')
# results2 = count_subset.total / g.total.transform('sum')