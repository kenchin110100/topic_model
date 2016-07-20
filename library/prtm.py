# coding: utf-8
"""
PageRankTopicModelのアルゴリズム
"""

from igraph import *
import collections
import numpy as np
from openopt import QP
import copy
from filer2.filer2 import Filer


class PRTM(object):

    def __init__(self, inputpath):
        # データを読み込んで、重み付き向こうグラフに変更する
        self._inputpath = inputpath
        if type(self._inputpath) == 'string':
            _list_edge = Filer.readcsv(self._inputpath)
        else:
            _list_edge = inputpath
        self._dict_network = self._cal_edgelist_to_network(_list_edge)
        # master用のネットワークを作成
        self._g_master = Graph()
        self._g_master.add_vertices(self._dict_network["vertex"])
        self._g_master.add_edges(self._dict_network["edge"])
        # 元のネットワークのpagerankを求める
        self._dict_network["pagerank"] = self._g_master.pagerank(directed=False, weights=self._dict_network["weight"])
        # ダミーのネットワーク、これを操作用とする
        self._dict_network_dammy = copy.deepcopy(self._dict_network)
        # 計算した後のサブネットワークの情報を記録するリスト
        self._list_dict_subnetwork = []
        # 計算した後の各トピックの単語の出現率を記録するリスト
        self._list_dict_word_prob = []
        # 各トピックの事前確率を記録するリスト
        self._list_topic_prob = []
        # クラスタリング後のノードを記録
        self._list_cluster_node = None
        # クラスタリングされた際のクラスタ数を記録
        self._K = None


    @property
    def dict_network(self):
        return self._dict_network

    @dict_network.setter
    def dict_network(self, value):
        self._dict_network = value

    @dict_network.getter
    def dict_network(self):
        return self._dict_network

    @dict_network.deleter
    def dict_network(self):
        del self._dict_network

    @property
    def list_dict_subnetwork(self):
        return self._list_dict_subnetwork

    @list_dict_subnetwork.setter
    def list_dict_subnetwork(self, value):
        self._list_dict_subnetwork = value

    @list_dict_subnetwork.getter
    def list_dict_subnetwork(self):
        return self._list_dict_subnetwork

    @list_dict_subnetwork.deleter
    def list_dict_subnetwork(self):
        del self._list_dict_subnetwork

    @property
    def list_dict_word_prob(self):
        return self._list_dict_word_prob

    @list_dict_word_prob.setter
    def list_dict_word_prob(self, value):
        self._list_dict_word_prob = value

    @list_dict_word_prob.getter
    def list_dict_word_prob(self):
        return self._list_dict_word_prob

    @list_dict_word_prob.deleter
    def list_dict_word_prob(self):
        del self._list_dict_word_prob

    @property
    def list_topic_prob(self):
        return self._list_topic_prob

    @list_topic_prob.setter
    def list_topic_prob(self, value):
        self._list_topic_prob = value

    @list_topic_prob.getter
    def list_topic_prob(self):
        return self._list_topic_prob

    @list_topic_prob.deleter
    def list_topic_prob(self):
        del self._list_topic_prob


    # 有向エッジリストを入力して、重み付き無向ネットワークを出力する
    def _cal_edgelist_to_network(self, list_edge):
        # 有向グラフならば
        if len(list_edge[0]) == 2:
            # 有向エッジリストを無向エッジリストに変換する
            list_edge = [tuple(sorted(row)) for row in list_edge]
            # ノードリスト
            list_vertices = list(set([word for row in list_edge for word in row]))
            # エッジリストとそのweightを作成
            tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())

            return {"vertex": list_vertices, "edge": list(tuple_edge), "weight": list(tuple_weight)}

        # 重み付き無向グラフならば
        if len(list_edge[0]) == 3:
            list_vertex = []
            list_edge_rev = []
            list_weight = []
            for row in list_edge:
                list_vertex.append(row[0])
                list_vertex.append(row[1])
                list_edge_rev.append([row[0], row[1]])
                list_weight.append(float(row[2]))
            list_vertex = list(set(list_vertex))

            return {"vertex": list_vertex, "edge": list_edge_rev, "weight": list_weight}

    def _cal_cluster_to_network(self, dict_network):
        if dict_network.has_key("cluster") == False:
            print "クラスタリングができていません"
            return []

        dict_cluster = collections.defaultdict(list)
        for word, cluster in zip(dict_network["vertex"], dict_network["cluster"]):
            dict_cluster[cluster].append(word)
        # リストに変換
        list_cluster_vertex = [row[1] for row in dict_cluster.items()]

        # 同様にエッジとウェイトのリストも作成する
        list_cluster_edge = []
        list_cluster_weight = []
        for cluster_vertex in list_cluster_vertex:
            list_cluster_edge_one = []
            list_cluster_weight_one = []
            # エッジリストの中に、一つでもノードが含まれていれば、そのクラスのノードに含める
            for row, weight in zip(dict_network["edge"], dict_network["weight"]):
                # and と or を切り替えることによって性能の比較
                if row[0] in cluster_vertex or row[1] in cluster_vertex:
                    list_cluster_edge_one.append(row)
                    list_cluster_weight_one.append(weight)
            list_cluster_edge.append(list_cluster_edge_one)
            list_cluster_weight.append(list_cluster_weight_one)

        # まとめる
        list_dict_subnetwork = [{"vertex": dict_network["vertex"],
                                 "edge": cluster_edge,
                                 "weight": cluster_weight}
                                for cluster_edge, cluster_weight
                                in zip(list_cluster_edge, list_cluster_weight)]

        return list_dict_subnetwork

    
    # 凸２次計画問題を解いてp(topic)を求めるための関数
    def _cal_prob_topic(self, dict_network, list_dict_subnetwork):
        prob_master = np.array([row[1] for row in
                                sorted(zip(dict_network["vertex"], dict_network["pagerank"]),
                                       key=lambda x: x[0])])

        for i, dict_subnetwork in enumerate(list_dict_subnetwork):
            if i == 0:
                prob_sub = np.array([row[1] for row in
                                     sorted(zip(dict_subnetwork["vertex"], dict_subnetwork["pagerank"]),
                                            key=lambda x: x[0])])
            else:
                list_tmp = np.array([row[1] for row in
                                     sorted(zip(dict_subnetwork["vertex"], dict_subnetwork["pagerank"]),
                                            key=lambda x: x[0])])
                prob_sub = np.vstack((prob_sub, list_tmp))

        H = 2 * prob_sub.dot(prob_sub.T)
        f = -2 * prob_master.dot(prob_sub.T)
        Aeq = np.ones(len(list_dict_subnetwork))
        beq = 1
        lb = np.zeros(len(list_dict_subnetwork))

        p = QP(H, f, Aeq=Aeq, beq=beq, lb=lb)
        r = p.solve("cvxopt_qp", iprint=-1)
        k_opt = r.xf
        return k_opt

    # 計算をするメインの関数
    def fit(self, n=0, flag_louvain=False, directed = False):
        # 内部変数の初期化
        self._list_dict_subnetwork = []
        self._dict_network_dammy = copy.deepcopy(self._dict_network)
        self._list_dict_word_prob = []
        self._list_topic_prob = []
        self._g_master = Graph()
        self._g_master.add_vertices(self._dict_network["vertex"])
        self._g_master.add_edges(self._dict_network["edge"])

        if flag_louvain:
            self._dict_network_dammy["cluster"] = self._g_master.community_multilevel(weights=self._dict_network_dammy["weight"]).membership
        else:
            if n == 0:
                self._dict_network_dammy["cluster"] = self._g_master.community_fastgreedy(weights=self._dict_network_dammy["weight"]).as_clustering().membership
            else:
                self._dict_network_dammy["cluster"] = self._g_master.community_fastgreedy(weights=self._dict_network_dammy["weight"]).as_clustering(n=n).membership

        # 元のネットワークのpagerankを求める
        self._dict_network_dammy["pagerank"] = self._g_master.pagerank(directed=directed, weights=self._dict_network_dammy["weight"])

        # クラスタ結果をもとにサブグラフのリストを作成
        self._list_dict_subnetwork = self._cal_cluster_to_network(self._dict_network_dammy)

        # サブクラスタごとに中心性の計算
        for i, dict_subnetwork in enumerate(self._list_dict_subnetwork):
            g_sub = Graph()
            g_sub.add_vertices(dict_subnetwork["vertex"])
            g_sub.add_edges(dict_subnetwork["edge"])
            self._list_dict_subnetwork[i]["pagerank"] = g_sub.pagerank(directed=directed, weights=dict_subnetwork["weight"])

        # トピックごとにwordを入力したらp(word|topic)が出るような辞書を作成
        for i in range(len(self._list_dict_subnetwork)):
            list_word_page = sorted(zip(self._list_dict_subnetwork[i]["vertex"], self._list_dict_subnetwork[i]["pagerank"]), key=lambda x: x[1], reverse=True)
            self._list_dict_word_prob.append({row[0]: row[1] for row in list_word_page})

        print "クラスタ数: ", len(self._list_dict_word_prob)
        self._list_topic_prob = self._cal_prob_topic(self._dict_network_dammy, self._list_dict_subnetwork)
        self._K = len(self._list_dict_word_prob)

    # 計算したトピックを表示するための関数
    def show_topic(self, n, num=10):
        print "トピック: ", n
        print "=============="
        for row in sorted(self.list_dict_word_prob[n].items(), key=lambda x:x[1], reverse=True)[0:num]:
            print row[0], row[1]
        print "=============="
    
    # クラスの推定
    def infer(self, list_words):
        try:
            # 学習が先に行われていなければエラーを上げる
            if self._list_topic_prob == None:
                raise NameError('calculation first')
            # すべての単語が辞書に含まれていなければエラーを上げる
            for word in list_words:
                if word in self._list_dict_word_prob[0]:
                    break
            else:
                raise KeyError('No word found in dict')
            # オーバーフロー対策
            list_overflow = []
            for i in range(self._K):
                prob = 0.0
                prob += np.log(self._list_topic_prob[i])
                for word in list_words:
                    if word in self._list_dict_word_prob[i]:
                        prob += np.log(self._list_dict_word_prob[i][word])
                list_overflow.append(prob)
            max_log = np.max(list_overflow)
            list_prob = np.array([np.exp(num - max_log) for num in list_overflow])
            # 正規化
            list_prob = list_prob/np.sum(list_prob)
            return list_prob
        
        except NameError:
            raise
        except KeyError:
            raise

class Evaluation(object):

    def __init__(self):
        # 結果の初期化
        self._dict_score = {}

    @property
    def dict_score(self):
        return self._dict_score

    @dict_score.setter
    def dict_score(self, value):
        self._dict_score = value

    @dict_score.getter
    def dict_score(self):
        return self._dict_score

    @dict_score.deleter
    def dict_score(self):
        del self._dict_score

    # f_measureを計算する
    def cal_f_measure(self, list_predict, list_measure):
        # 内部変数の初期化
        self._dict_score = {}

        # 生成したクラスタ内のカウント
        dict_predict_cluster = collections.defaultdict(list)
        for row in zip(list_predict, list_measure):
            dict_predict_cluster[row[0]].append(row[1])

        # もとあるクラス内のカウント
        dict_measure_cluster = collections.defaultdict(list)
        for row in zip(list_predict, list_measure):
            dict_measure_cluster[row[1]].append(row[0])

        # local_purityの計算
        list_purity = []
        for row in dict_predict_cluster.items():
            major_class = sorted(collections.Counter(row[1]).items(), key=lambda x: x[1], reverse=True)[0][1]
            class_num = len(row[1])
            list_purity.append([major_class, class_num])
        purity = float(np.sum(zip(*list_purity)[0])) / np.sum(zip(*list_purity)[1])

        # inverse_purityの計算
        list_inverse_purity = []
        for row in dict_measure_cluster.items():
            major_class = sorted(collections.Counter(row[1]).items(), key=lambda x: x[1], reverse=True)[0][1]
            class_num = len(row[1])
            list_inverse_purity.append([major_class, class_num])
        inverse_purity = float(np.sum(zip(*list_inverse_purity)[0])) / np.sum(zip(*list_inverse_purity)[1])

        self._dict_score = {"purity": purity, "invpurity":inverse_purity, "fvalue":2/(1/purity+1/inverse_purity), 'K': len(set(list_predict))}
        return self._dict_score

    # 計算したスコアを表示する関数
    def show_score(self):
        if self._dict_score == {}:
            print "まだ計算が行われていません"
        else:
            print "============"
            print "Purity: ", self._dict_score["purity"]
            print "InversePurity: ", self._dict_score["invpurity"]
            print "F-value: ", self._dict_score["fvalue"]
            print "============"
