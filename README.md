# Cosine_Similarity_Recommend
基于用户余弦相似度的电影（商品）推荐

## 输入
输入带有以下信息的csv文件
```
1. 用户id
2. 电影（或商品）id
3. 用户对电影（或商品）打分
```
如：ratings.csv  （该数据集来自 https://grouplens.org/datasets/movielens/ ）

## 输出
输出推荐结果（一个json文件，key为用户id，value为向该用户推荐的电影（或商品）id）
