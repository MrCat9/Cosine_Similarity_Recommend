# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import json


class CosineSimilarityRecommend(object):
    def __init__(self, csv_path, user_id_item, movie_id_item, rating_item, min_seen_same_movies_number, min_cosine_similarity, min_recommend_rating):
        self.csv_path = csv_path  # csv 文件路径
        self.user_id_item = user_id_item  # csv 文件中用户id的字段名
        self.movie_id_item = movie_id_item  # csv 文件中电影id的字段名
        self.rating_item = rating_item  # csv 文件中评分的字段名
        self.MIN_SEEN_SAME_MOVIES_NUMBER = min_seen_same_movies_number  # 两个用户至少看过 X 部相同的电影，才计算他们之间的余弦相似度
        self.MIN_COSINE_SIMILARITY = min_cosine_similarity  # 两个用户之间的余弦相似度至少为 X 时，才让这两个用户相互推荐电影
        self.MIN_RECOMMEND_RATING = min_recommend_rating  # 两个相似的用户，user1和user2，user2只向user1推荐user2评分高于 X 的电影

        self.df = ''  # 从csv文件读取出来的内容的DataFrame

        self.users_set = ''  # 用户集合
        self.movies_set = ''  # 电影集合
        # self.users_movies_ndarray = ''  # 用户_电影 矩阵，行为每个用户的索引，列为每部电影的索引（索引值从0开始）

        self.users_combinations_df = ''  # 用户两两组合的DataFrame

        self.movies_seen_users_set_dict = ''  # 每部电影的用户集合的字典
        # {
        #     movie_id_0: {user_id_0, user_id_1},  # key_type:int
        #     movie_id_1: {user_id_0, user_id_4},  # value_type:set    {int, int, int}
        # }

        self.need_recommend_users_seen_movies_set_dict = ''  # 每个需要推荐的用户看过的电影集合

        self.recommend_result_dict = ''  # 推荐结果
        # {
        #     user_id_0: "movie_id_0, movie_id_2",  # key_type:int  value_type:str
        #     user_id_1: "movie_id_1",
        #     user_id_2: "movie_id_0, movie_id_1, movie_id_2"
        # }

    @staticmethod
    def save_dict_to_json(dict_data, json_path='recommend_result.json'):
        json_str = json.dumps(dict_data, indent=4)
        with open(json_path, 'w') as json_file:
            json_file.write(json_str)

    @staticmethod
    def make_dict_data_serializable(dict_data):
        new_dict_data = {}
        for _key, _value in dict_data.items():  # 把 dict 中的 value 都转成 str
            if _value:
                new_dict_data[str(_key)] = str(_value)[1:-1]  # [1:-1]去掉set的前后花括号{}
            else:
                new_dict_data[str(_key)] = ''
        return new_dict_data

    # def recommend_for_users(self, user1_id, user2_id):
    #     pass

    def generate_users_seen_movies_set_dict(self, users_set, df, user_id_item, movie_id_item):
        """
        生成每个需要推荐的用户看过的电影集合
        :param users_set:
        :param df:
        :param user_id_item:
        :param movie_id_item:
        :return:
        """
        users_seen_movies_set_dict = {}
        for _user in users_set:
            # temp_df = df[df[user_id_item] == _user]  # 取出 用户_user 的所有行
            temp_df = df.loc[df.loc[:, user_id_item] == _user]  # 取出 用户_user 的所有行
            movies_set = set(temp_df.loc[:, movie_id_item])  # 取出 用户_user 看过的所有电影
            users_seen_movies_set_dict[_user] = movies_set  # 存入 dict
        self.need_recommend_users_seen_movies_set_dict = users_seen_movies_set_dict
        return self.need_recommend_users_seen_movies_set_dict

    def generate_recommend_result_dict(self, need_recommend_df, df, user_id_item, movie_id_item, rating_item, min_recommend_rating):
        """
        生成推荐结果 dict
        {
            user_id_0: "movie_id_0, movie_id_2",  # key_type:int  value_type:str
            user_id_1: "movie_id_1",
            user_id_2: "movie_id_0, movie_id_1, movie_id_2"
        }
        :param need_recommend_df:
        :param df:
        :param user_id_item:
        :param movie_id_item:
        :param rating_item:
        :param min_recommend_rating:
        :return:
        """
        # users_set = set()  # 需要推荐的用户集合
        # users_df = need_recommend_df.loc[:, ['user1', 'user2']].applymap(lambda x: users_set.add(x))
        user1_set = set(need_recommend_df.loc[:, 'user1'])
        user2_set = set(need_recommend_df.loc[:, 'user2'])
        users_set = user1_set.union(user2_set)  # 并集  # 需要推荐的用户集合
        need_recommend_users_seen_movies_set_dict = self.generate_users_seen_movies_set_dict(users_set, df, user_id_item, movie_id_item)

        # 推荐
        recommend_result_dict = {}
        for _row in need_recommend_df.itertuples():
            user1_id = getattr(_row, 'user1')
            user2_id = getattr(_row, 'user2')
            user1_seen_movies_set = need_recommend_users_seen_movies_set_dict[user1_id]
            user2_seen_movies_set = need_recommend_users_seen_movies_set_dict[user2_id]
            empty_set = set()

            # ======== 给user1推荐 ========
            recommend_for_users1_set = set()
            temp_set = user2_seen_movies_set.difference(user1_seen_movies_set)  # u2 - u1
            for _movie_id in temp_set:
                # rating = df[(df[movie_id_item] == _movie_id) & (df[user_id_item] == user2_id)].loc[:, rating_item]
                # rating = df.loc[(df.loc[:, movie_id_item] == _movie_id) & (df.loc[:, user_id_item] == user2_id)].loc[:, rating_item]
                rating = df.loc[(df.loc[:, movie_id_item] == _movie_id) & (df.loc[:, user_id_item] == user2_id), rating_item]
                float_rating = rating.iat[0]
                if float_rating >= min_recommend_rating:
                    recommend_for_users1_set.add(_movie_id)
            old_recommend_set = recommend_result_dict.get(user1_id, empty_set)
            recommend_result_dict[user1_id] = recommend_for_users1_set.union(old_recommend_set)  # 并集
            # =============================

            # ======== 给user2推荐 ========
            recommend_for_users2_set = set()
            temp_set = user1_seen_movies_set.difference(user2_seen_movies_set)
            for _movie_id in temp_set:
                # rating = df[(df[movie_id_item] == _movie_id) & (df[user_id_item] == user1_id)].loc[:, rating_item]
                # rating = df.loc[(df.loc[:, movie_id_item] == _movie_id) & (df.loc[:, user_id_item] == user1_id)].loc[:, rating_item]
                rating = df.loc[(df.loc[:, movie_id_item] == _movie_id) & (df.loc[:, user_id_item] == user1_id), rating_item]
                float_rating = rating.iat[0]
                if float_rating >= min_recommend_rating:
                    recommend_for_users2_set.add(_movie_id)
            old_recommend_set = recommend_result_dict.get(user2_id, empty_set)
            recommend_result_dict[user2_id] = recommend_for_users2_set.union(old_recommend_set)
            # =============================

        recommend_result_dict = self.make_dict_data_serializable(recommend_result_dict)
        self.recommend_result_dict = recommend_result_dict
        return self.recommend_result_dict

    def filter_need_recommend_df(self, users_cosine_similarity_df, min_cosine_similarity):
        """
        过滤出需要推荐的 DataFrame
        根据两个用户之间的余弦相似度进行过滤
        过滤余弦相似度小于 min_cosine_similarity 的用户组合
        :param users_cosine_similarity_df:
        :param min_cosine_similarity:
        :return:
        """
        # need_recommend_df = users_cosine_similarity_df[users_cosine_similarity_df['cosine_similarity'] >= min_cosine_similarity]
        need_recommend_df = users_cosine_similarity_df.loc[users_cosine_similarity_df.loc[:, 'cosine_similarity'] >= min_cosine_similarity]
        self.users_combinations_df = need_recommend_df
        return self.users_combinations_df

    @staticmethod
    def calculate_vectors_cosine_similarity(vector1, vector2):
        """
        计算两个向量的夹角余弦
        :param vector1: 列向量
        :param vector2: 列向量
        :return:
        """
        v1 = np.array(vector1).reshape(1, -1)
        v2 = np.array(vector2).reshape(1, -1)
        try:
            cos_sim = cosine_similarity(X=v1, Y=v2)
            # 两个向量计算余弦相似度时，计算结果是一个 1*1矩阵
            if cos_sim.size == 1:
                cos_sim = cos_sim[0, 0]
            else:
                cos_sim = 0.0
            return cos_sim
        except Exception as e:  # 可能是两个空向量
            print(str(e))
            return 0.0

    # @staticmethod
    # def vectors_dimensionality_reduction(vector1, vector2):
    #     """
    #     对传入的两个向量进行降维，返回两个降维后的新向量
    #     :param vector1:
    #     :param vector2:
    #     :return:
    #     """
    #     new_v1_list = []  # 新向量1
    #     new_v2_list = []
    #     number = vector1.size  # 大小，看要迭代几次
    #     for i in range(number):
    #         vector1_i = vector1[i]
    #         vector2_i = vector2[i]
    #         if vector1_i != 0.0 or vector2_i != 0.0:  # 两向量对应位置不同时为0时，保留该项
    #             new_v1_list.append(vector1_i)
    #             new_v2_list.append(vector2_i)
    #     new_v1 = np.array(new_v1_list)  # 生成新向量1
    #     new_v2 = np.array(new_v2_list)
    #     return new_v1, new_v2

    def calculate_users_cosine_similarity(self, need_calculate_df, df, user_id_item, movie_id_item, rating_item):
        """
        计算用户相似度
        :param need_calculate_df:
        :return:
        """
        cos_sim_list = []  # 两两用户的相似度  # [0.6044, 0.6212, 0.6003]
        for _row in need_calculate_df.itertuples():
            # user1_index = getattr(_row, 'user1') - 1  # 要 -1 ，因为矩阵的索引是从0开始的
            # user2_index = getattr(_row, 'user2') - 1  # 要 -1 ，因为矩阵的索引是从0开始的
            # user1_vector = users_movies_ndarray[user1_index]  # 取行  # 用户向量
            # user2_vector = users_movies_ndarray[user2_index]  # shape: (50006,)  # 50006 为电影数量
            # user1_vector, user2_vector = self.vectors_dimensionality_reduction(user1_vector, user2_vector)  # 对向量降维
            user1_id = getattr(_row, 'user1')
            user2_id = getattr(_row, 'user2')
            # user1_df = df.loc[df.loc[:, user_id_item] == user1_id].loc[:, [movie_id_item, rating_item]]
            user1_df = df.loc[df.loc[:, user_id_item] == user1_id, [movie_id_item, rating_item]]
            # user2_df = df.loc[df.loc[:, user_id_item] == user2_id].loc[:, [movie_id_item, rating_item]]
            user2_df = df.loc[df.loc[:, user_id_item] == user2_id, [movie_id_item, rating_item]]
            user12_df = pd.merge(user1_df, user2_df, on=movie_id_item, how='outer').fillna(value=0)
            user1_vector = getattr(user12_df, rating_item+'_x')
            user2_vector = getattr(user12_df, rating_item+'_y')
            cos_sim = self.calculate_vectors_cosine_similarity(user1_vector, user2_vector)  # 传入列向量  # 计算两个向量的夹角余弦
            cos_sim_list.append(cos_sim)  # 将该对用户的余弦相似度保存到list里
        users_cosine_similarity_df = need_calculate_df
        users_cosine_similarity_df.insert(loc=0, column='cosine_similarity', value=cos_sim_list)  # DataFrame 拼接
        self.users_combinations_df = users_cosine_similarity_df
        return self.users_combinations_df

    def filter_need_calculate_df(self, two_users_seen_same_movies_df, min_seen_same_movies_number):
        """
        过滤出需要计算相似度的 DataFrame
        根据两个用户看过的相同电影的数量进行过滤
        过滤相同电影数量小于 min_seen_same_movies_number 的用户组合
        :param two_users_seen_same_movies_df:
        :param min_seen_same_movies_number:
        :return:
        """
        # need_calculate_df = two_users_seen_same_movies_df[two_users_seen_same_movies_df['seen_same_movies_number'] >= min_seen_same_movies_number]
        need_calculate_df = two_users_seen_same_movies_df.loc[two_users_seen_same_movies_df.loc[:, 'seen_same_movies_number'] >= min_seen_same_movies_number]
        self.users_combinations_df = need_calculate_df
        return self.users_combinations_df

    def count_seen_same_movies(self, users_combinations_df, movies_seen_users_set_dict):
        """
        计算两两用户看过相同电影的数量
        :param users_combinations_df:
        :param movies_seen_users_set_dict:
        :return:
        """
        seen_same_movies_id_list = []  # 两两用户看过的相同电影的id  # ['1, 2', '', '1, 3, 4']
        seen_same_movies_number_list = []  # 两两用户看过的相同电影的数量  # [2, 0, 3]
        for _row in users_combinations_df.itertuples():  # 迭代出每个用户组合
            # users_combinations_set = set(_row[1:])  # 每行的第一个为 index,所以用[1:]去掉
            seen_same_movie_set = set()
            for _key, _value in movies_seen_users_set_dict.items():  # 迭代出每一部电影，检查两人看过哪些相同的电影
                if set(_row[1:]).issubset(_value):  # 两人看过相同的电影，该电影的id为_key
                    seen_same_movie_set.add(_key)
            # if seen_same_movie_set:
            #     seen_same_movies_id = str(seen_same_movie_set)[1:-1]  # [1:-1]去掉set的前后花括号{}
            # else:
            #     seen_same_movies_id = ''
            seen_same_movie_list = list(seen_same_movie_set)
            seen_same_movie_list.sort()  # list 排序
            seen_same_movies_id = str(seen_same_movie_list).replace('[', '').replace(']', '')  # str
            seen_same_movies_number = len(seen_same_movie_set)  # int

            seen_same_movies_id_list.append(seen_same_movies_id)
            seen_same_movies_number_list.append(seen_same_movies_number)
            print(seen_same_movies_number, seen_same_movies_id)
        two_users_seen_same_movies_df = users_combinations_df
        two_users_seen_same_movies_df.insert(loc=0, column='seen_same_movies_id', value=seen_same_movies_id_list)  # DataFrame 列拼接
        two_users_seen_same_movies_df.insert(loc=0, column='seen_same_movies_number', value=seen_same_movies_number_list)
        self.users_combinations_df = two_users_seen_same_movies_df
        return self.users_combinations_df

    def generate_movies_seen_users_set_dict(self, df, user_id_item, movie_id_item):
        """
        生成每部电影的用户集合的字典
        {
            movie_id_0: {user_id_0, user_id_1},  # key_type:int
            movie_id_1: {user_id_0, user_id_4},  # value_type:set    {int, int, int}
        }
        :param df:
        :param user_id_item:
        :param movie_id_item:
        :return:
        """
        movies_set = set(df.loc[:, movie_id_item])
        self.movies_set = movies_set

        movies_seen_users_set_dict = {}
        for _movie in movies_set:  # 迭代出每一个电影id
            # temp_df = df[df[movie_id_item] == _movie]  # 取出看过 _movie 这部电影的所有行  # DataFrame 的布尔索引
            temp_df = df.loc[df.loc[:, movie_id_item] == _movie]  # 取出看过 _movie 这部电影的所有行  # DataFrame 的布尔索引
            users_set = set(temp_df.loc[:, user_id_item])  # 取出看过 _movie 这部电影的所有用户
            movies_seen_users_set_dict[_movie] = users_set  # 存入 dict
        self.movies_seen_users_set_dict = movies_seen_users_set_dict
        return self.movies_set, self.movies_seen_users_set_dict

    def generate_users_combinations_df(self, df, user_id_item):
        """
        生成所有用户的两两组合的 DataFrame
        :param users_set:
        :return:
        """
        users_set = set(df.loc[:, user_id_item])
        self.users_set = users_set

        users_combinations = itertools.combinations(list(users_set), 2)  # 组合
        self.users_combinations_df = pd.DataFrame(data=list(users_combinations), columns=['user1', 'user2'])  # 所有用户的两两组合

        return self.users_set, self.users_combinations_df

    # def generate_users_movies_ndarray(self, df, user_id_item, movie_id_item, rating_item):
    #     """
    #     生成 用户_电影 矩阵，行为每个用户的索引，列为每部电影的索引（索引值从0开始）
    #     矩阵中的值为该用户对该电影的评分
    #     :param df:
    #     :param user_id_item:
    #     :param movie_id_item:
    #     :param rating_item:
    #     :return:
    #     """
    #     # self.users_set = set(df.loc[:, user_id_item].drop_duplicates())
    #     # self.users_set = set(df.drop_duplicates([user_id_item]).loc[:, user_id_item])  # drop_duplicates()  对列去重
    #     self.users_set = set(df.loc[:, user_id_item])
    #     self.movies_set = set(df.loc[:, movie_id_item])
    #     self.users_movies_ndarray = np.zeros((max(self.users_set), max(self.movies_set)))
    #     for _row in df.itertuples():
    #         # print(getattr(row, 'userId'), getattr(row, 'movieId'), getattr(row, 'rating'))
    #         # print(_row.userId)
    #         user_index = getattr(_row, user_id_item) - 1  # 要 -1 ，因为矩阵的索引是从0开始的
    #         movie_index = getattr(_row, movie_id_item) - 1
    #         rating = getattr(_row, rating_item)
    #         self.users_movies_ndarray[user_index, movie_index] = rating
    #     return self.users_set, self.movies_set, self.users_movies_ndarray

    def read_csv_to_df(self, csv_path):
        self.df = pd.read_csv(csv_path)  # 读csv
        return self.df

    def main(self):
        # 读 csv
        df = self.read_csv_to_df(self.csv_path)
        # 生成 用户_电影 矩阵
        # users_set, movies_set, users_movies_ndarray = self.generate_users_movies_ndarray(self.df, self.user_id_item, self.movie_id_item, self.rating_item)
        # 生成所有用户的两两组合的 DataFrame
        users_set, users_combinations_df = self.generate_users_combinations_df(self.df, self.user_id_item)
        # 生成每部电影的用户集合的字典
        movies_set, movies_seen_users_set_dict = self.generate_movies_seen_users_set_dict(self.df, self.user_id_item, self.movie_id_item)
        # 计算两两用户看过相同电影的数量
        two_users_seen_same_movies_df = self.count_seen_same_movies(self.users_combinations_df, self.movies_seen_users_set_dict)
        # 过滤出需要计算相似度的 DataFrame
        need_calculate_df = self.filter_need_calculate_df(self.users_combinations_df, self.MIN_SEEN_SAME_MOVIES_NUMBER)
        # 计算用户相似度
        users_cosine_similarity_df = self.calculate_users_cosine_similarity(self.users_combinations_df, self.df, self.user_id_item, self.movie_id_item, self.rating_item)
        # 过滤出需要推荐的 DataFrame
        need_recommend_df = self.filter_need_recommend_df(self.users_combinations_df, self.MIN_COSINE_SIMILARITY)
        # 生成推荐结果 dict
        recommend_result_dict = self.generate_recommend_result_dict(self.users_combinations_df, self.df, self.user_id_item, self.movie_id_item, self.rating_item, self.MIN_RECOMMEND_RATING)
        # 将 dict 数据保存到json文件中
        self.save_dict_to_json(self.recommend_result_dict)


if __name__ == '__main__':
    # give_csv_path = 'ratings_temp.csv'
    give_csv_path = 'ratings.csv'
    give_user_id_item = 'userId'
    give_movie_id_item = 'movieId'
    give_rating_item = 'rating'
    give_min_seen_same_movies_number = 20
    give_min_cosine_similarity = 0.5
    give_min_recommend_rating = 4.0
    cosine_similarity_recommend = CosineSimilarityRecommend(give_csv_path, give_user_id_item, give_movie_id_item, give_rating_item, give_min_seen_same_movies_number, give_min_cosine_similarity, give_min_recommend_rating)
    cosine_similarity_recommend.main()
    print('='*16, 'end', '='*16)

    # give_csv_path = 'test_csv.csv'
    # give_user_id_item = 'userid'
    # give_movie_id_item = 'goodsid'
    # give_rating_item = 'score'
    # give_min_seen_same_movies_number = 2
    # give_min_cosine_similarity = 0.6
    # give_min_recommend_rating = 4.0
    # cosine_similarity_recommend = CosineSimilarityRecommend(give_csv_path, give_user_id_item, give_movie_id_item, give_rating_item, give_min_seen_same_movies_number, give_min_cosine_similarity, give_min_recommend_rating)
    # cosine_similarity_recommend.main()
    # print('='*16, 'end', '='*16)
