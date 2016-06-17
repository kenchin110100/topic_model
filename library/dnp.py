# coding: utf-8
"""
divide_newman_pagerankのアルゴリズム
"""

from igraph import *
import collections
import numpy as np
from openopt import QP
import copy
import re
from filer import Filer


class DNP(object):

    def __init__(self, inputpath):
        # データを読み込んで、重み付き向こうグラフに変更する
        self._inputpath = inputpath
        if type(self._inputpath) == 'string':
            _list_edge = Filer.readcsv(self._inputpath)
        else:
            _list_edge = inputpath
        self._dict_network_master = self._cal_edgelist_to_network(_list_edge)
        # master用のネットワークを作成
        self._g_master = Graph()
        self._g_master.add_vertices(self._dict_network_master["vertex"])
        self._g_master.add_edges(self._dict_network_master["edge"])
        # 元のネットワークのpagerankを求める
        self._dict_network_master["pagerank"] = self._g_master.pagerank(directed=False, weights=self._dict_network_master["weight"])
        # ダミーのネットワーク、これを操作用とする
        self._dict_network_master_dammy = copy.deepcopy(self._dict_network_master)
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
    def dict_network_master(self):
        return self._dict_network_master

    @dict_network_master.setter
    def dict_network_master(self, value):
        self._dict_network_master = value

    @dict_network_master.getter
    def dict_network_master(self):
        return self._dict_network_master

    @dict_network_master.deleter
    def dict_network_master(self):
        del self._dict_network_master

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

    def _cal_cluster_to_network(self, dict_network, flag_OPIC):
        if dict_network.has_key("cluster") == False:
            print "クラスタリングができていません"
            return []

        if flag_OPIC == True:
            list_cluster_vertex = self._cal_opic(dict_network, threshold=self.threshold, iteration=self.iteration)
        # クラスタごとにwordをまとめる
        else:
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

    # OPICアプローチの計算
    def _cal_opic(self, dict_network, threshold, iteration):
        # ノード、エッジ、ウェイト、クラスタのリストの作成
        list_vertex = dict_network['vertex']
        list_cluster = dict_network['cluster']
        list_weight = dict_network['weight']
        list_edge = dict_network['edge']
        # クラスタ数の記録
        K = len(set(list_cluster))

        # クラスタごとのノード、ノードに対応するクラスタを返す辞書の作成
        list_cluster_node = [[] for i in range(K)]
        for cluster, vertex in zip(list_cluster, list_vertex):
            list_cluster_node[cluster].append(vertex)
        dict_node_cluster = {node:[i] for i, row in enumerate(list_cluster_node) for node in row}

        # ここより以下が再計算の対象
        for i in range(iteration):
            dict_node_weight = self._cal_opic_iteration(list_cluster_node, dict_node_cluster, K, list_weight, list_edge, list_vertex)
            # 閾値をもとに、新しく割り当てられたクラスタに追加する
            for node, row in dict_node_weight.items():
                for index in np.where(dict_node_weight[node] > threshold)[0]:
                    if node not in list_cluster_node[index]:
                        list_cluster_node[index].append(node)
                        dict_node_cluster[node].append(index)

        return list_cluster_node

    # OPICアプローチの中で再帰的の計算される部分
    def _cal_opic_iteration(self, list_cluster_node, dict_node_cluster, K, list_weight, list_edge, list_vertex):
        # 各ノードが各クラスタに対して持っているノード数を記録
        dict_node_weight = {node: np.array([0.0 for i in range(K)]) for node in list_vertex}
        # エッジを計算
        for edge, weight in zip(list_edge, list_weight):
            for c in dict_node_cluster[edge[0]]:
                dict_node_weight[edge[1]][c] += weight
            for c in dict_node_cluster[edge[1]]:
                dict_node_weight[edge[0]][c] += weight
        for key, row in dict_node_weight.items():
            dict_node_weight[key] /= np.sum(row)

        return dict_node_weight

    # 凸２次計画問題を解いてp(topic)を求めるための関数
    def _cal_prob_topic(self, dict_network_master, list_dict_subnetwork):
        prob_master = np.array([row[1] for row in
                                sorted(zip(dict_network_master["vertex"], dict_network_master["pagerank"]),
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
        r = p.solve("cvxopt_qp")
        k_opt = r.xf
        return k_opt

    # クラスタリング時に条件にあったノードを分割する
    def _cal_divide_node(self, dict_network, low_fleq=0.05, low_rate=0.8, flag=0):
        if dict_network.has_key("cluster") == False:
            print "クラスタリングができていません"

        # wordをclusterに変換するための辞書を作成する
        dict_word_to_cluster = {}
        for word, cluster in zip(dict_network["vertex"], dict_network["cluster"]):
            dict_word_to_cluster[word] = cluster

        # wordをidに変換するための辞書を作成する
        dict_word_to_id = {}
        for i, word in enumerate(dict_network["vertex"]):
            dict_word_to_id[word] = i

        # clusterの数
        num_cluster = len(set(dict_network["cluster"]))

        # id_to_clusterのマトリックス
        matrix_id_to_cluster = np.zeros((len(dict_word_to_id), num_cluster))
        for row, weight in zip(dict_network["edge"], dict_network["weight"]):
            matrix_id_to_cluster[dict_word_to_id[row[0]]][dict_word_to_cluster[row[1]]] += weight
            matrix_id_to_cluster[dict_word_to_id[row[1]]][dict_word_to_cluster[row[0]]] += weight

        # 総単語数を求める
        total_voc = np.sum(matrix_id_to_cluster)

        # 指定したハイパーパラメータよりも高い値を記録した単語を所属クラスタとともに辞書に登録する
        dict_word_to_list_cluster = {}
        for cluster, word, row in zip(dict_network["cluster"], dict_network["vertex"], matrix_id_to_cluster):
            top_num = max(row)
            if (float(np.sum(row))/total_voc)>=low_fleq and len(np.where(row/top_num>=low_rate)[0])>=2:
                dict_word_to_list_cluster[word] = np.where(row / top_num >= low_rate)[0]

        # 分割する単語が存在するかしないかflagを立てる
        if len(dict_word_to_list_cluster)>0:
            outflag = True
        else:
            outflag = False

        # 新しく分裂するノードクラスターを元のクラスターに記録
        for word, row in dict_word_to_list_cluster.items():
            for num in row:
                dict_word_to_cluster[word + "_" + str(num)] = dict_word_to_cluster[word]

        # ここの計算が間違ってそう => diffsplitで確認したが大丈夫そう・・・
        for word in dict_word_to_list_cluster.keys():
            list_edge_new = []
            list_weight_new = []
            for row, weight in zip(dict_network["edge"], dict_network["weight"]):
                if row[0] == word:
                    if dict_word_to_cluster[row[1]] in dict_word_to_list_cluster[word]:
                        list_edge_new.append([row[0] + "_" + str(dict_word_to_cluster[row[1]]), row[1]])
                        list_weight_new.append(weight)
                    else:
                        for num in dict_word_to_list_cluster[word]:
                            # flagの値によって、weightの計算法を切り替え
                            # 0なら小数あり
                            if flag == 0:
                                weight_tmp = float(weight) / len(dict_word_to_list_cluster[word])
                                list_edge_new.append([row[0] + "_" + str(num), row[1]])
                                list_weight_new.append(weight_tmp)
                            # 1なら小数は切り捨て
                            else:
                                weight_tmp = weight / len(dict_word_to_list_cluster[word])
                                if weight_tmp > 0:
                                    list_edge_new.append([row[0] + "_" + str(num), row[1]])
                                    list_weight_new.append(weight_tmp)

                elif row[1] == word:
                    if dict_word_to_cluster[row[0]] in dict_word_to_list_cluster[word]:
                        list_edge_new.append([row[0], row[1] + "_" + str(dict_word_to_cluster[row[0]])])
                        list_weight_new.append(weight)
                    else:
                        for num in dict_word_to_list_cluster[word]:
                            # flagの値によって、weightの計算法を切り替え
                            # 0なら小数あり
                            if flag == 0:
                                weight_tmp = float(weight) / len(dict_word_to_list_cluster[word])
                                list_edge_new.append([row[0], row[1] + "_" + str(num)])
                                list_weight_new.append(weight_tmp)
                            # 1なら小数は切り捨て
                            else:
                                weight_tmp = weight / len(dict_word_to_list_cluster[word])
                                if weight_tmp > 0:
                                    list_edge_new.append([row[0], row[1] + "_" + str(num)])
                                    list_weight_new.append(weight_tmp)
                else:
                    list_edge_new.append([row[0], row[1]])
                    list_weight_new.append(weight)
            else:
                dict_network["edge"] = copy.deepcopy(list_edge_new)
                dict_network["weight"] = copy.deepcopy(list_weight_new)
                list_vertices = list(set([word for row in dict_network["edge"] for word in row]))
                dict_network["vertex"] = list_vertices

        return dict_network, outflag

    def _cal_DN_cluster(self, low_freq, low_rate, n, flag_louvain):
        if flag_louvain:
            outflag = True
            while outflag:
                # louvain法によるクラスタリング、vertexと同じ長さのクラスタ番号が書かれたリストがreturn
                self._dict_network_master_dammy["cluster"] = self._g_master.community_multilevel(weights=self._dict_network_master_dammy["weight"]).membership
                self._dict_network_master_dammy, outflag = self._cal_divide_node(self._dict_network_master_dammy, low_fleq=low_freq, low_rate=low_rate, flag=0)
                self._g_master = Graph()
                self._g_master.add_vertices(self._dict_network_master_dammy["vertex"])
                self._g_master.add_edges(self._dict_network_master_dammy["edge"])
            self._dict_network_master_dammy["cluster"] = self._g_master.community_multilevel(weights=self._dict_network_master_dammy["weight"]).membership
            self._dict_network_master_dammy["pagerank"] = self._g_master.pagerank(directed=False, weights=self._dict_network_master_dammy["weight"])

        else:
            if n == 0:
                outflag = True
                while outflag:
                    # louvain法によるクラスタリング、vertexと同じ長さのクラスタ番号が書かれたリストがreturn
                    self._dict_network_master_dammy["cluster"] = self._g_master.community_fastgreedy(weights=self._dict_network_master_dammy["weight"]).as_clustering().membership
                    self._dict_network_master_dammy, outflag = self._cal_divide_node(self._dict_network_master_dammy, low_fleq=low_freq, low_rate=low_rate, flag=0)
                    self._g_master = Graph()
                    self._g_master.add_vertices(self._dict_network_master_dammy["vertex"])
                    self._g_master.add_edges(self._dict_network_master_dammy["edge"])
                self._dict_network_master_dammy["cluster"] = self._g_master.community_fastgreedy(weights=self._dict_network_master_dammy["weight"]).as_clustering().membership
            else:
                outflag = True
                while outflag:
                    # louvain法によるクラスタリング、vertexと同じ長さのクラスタ番号が書かれたリストがreturn
                    self._dict_network_master_dammy["cluster"] = self._g_master.community_fastgreedy(weights=self._dict_network_master_dammy["weight"]).as_clustering(n=n).membership
                    self._dict_network_master_dammy, outflag = self._cal_divide_node(self._dict_network_master_dammy, low_fleq=low_freq, low_rate=low_rate, flag=0)
                    self._g_master = Graph()
                    self._g_master.add_vertices(self._dict_network_master_dammy["vertex"])
                    self._g_master.add_edges(self._dict_network_master_dammy["edge"])
                self._dict_network_master_dammy["cluster"] = self._g_master.community_fastgreedy(weights=self._dict_network_master_dammy["weight"]).as_clustering(n=n).membership



    # 計算をするメインの関数
    def calculation(self, n=0, flag_louvain=False, flag_OPIC=False, threshold=0.3, iteration=1, directed = False):
        # DNをするフラッグ（多分することはないから中だし）
        flag_DN = False
        # 内部変数の初期化
        self.threshold = threshold
        self.iteration = iteration
        self._list_dict_subnetwork = []
        self._dict_network_master_dammy = copy.deepcopy(self._dict_network_master)
        self._list_dict_word_prob = []
        self._list_topic_prob = []
        self._g_master = Graph()
        self._g_master.add_vertices(self._dict_network_master["vertex"])
        self._g_master.add_edges(self._dict_network_master["edge"])

        if flag_DN == True:
            self._cal_DN_cluster(low_freq=low_freq, low_rate=low_rate, n=n, flag_louvain=flag_louvain)
        else:
            if flag_louvain:
                self._dict_network_master_dammy["cluster"] = self._g_master.community_multilevel(weights=self._dict_network_master_dammy["weight"]).membership
            else:
                if n == 0:
                    self._dict_network_master_dammy["cluster"] = self._g_master.community_fastgreedy(weights=self._dict_network_master_dammy["weight"]).as_clustering().membership
                else:
                    self._dict_network_master_dammy["cluster"] = self._g_master.community_fastgreedy(weights=self._dict_network_master_dammy["weight"]).as_clustering(n=n).membership

        # 元のネットワークのpagerankを求める
        self._dict_network_master_dammy["pagerank"] = self._g_master.pagerank(directed=directed, weights=self._dict_network_master_dammy["weight"])

        # クラスタ結果をもとにサブグラフのリストを作成
        self._list_dict_subnetwork = self._cal_cluster_to_network(self._dict_network_master_dammy, flag_OPIC)

        # サブクラスタごとに中心性の計算
        for i, dict_subnetwork in enumerate(self._list_dict_subnetwork):
            g_sub = Graph()
            g_sub.add_vertices(dict_subnetwork["vertex"])
            g_sub.add_edges(dict_subnetwork["edge"])
            self._list_dict_subnetwork[i]["pagerank"] = g_sub.pagerank(directed=directed, weights=dict_subnetwork["weight"])

        # トピックごとにwordを入力したらp(word|topic)が出るような辞書を作成
        for i in range(len(self._list_dict_subnetwork)):
            list_word_page = sorted(zip(self._list_dict_subnetwork[i]["vertex"], self._list_dict_subnetwork[i]["pagerank"]), key=lambda x: x[1], reverse=True)
            list_word_page_rev = []
            for row in list_word_page:
                pattern = "_[0-9|_]+"
                word = re.sub(pattern, "", row[0])
                list_word_page_rev.append([word, row[1]])
            self._list_dict_word_prob.append({row[0]: row[1] for row in list_word_page_rev})

        print "クラスタ数: ", len(self._list_dict_word_prob)
        self._list_topic_prob = self._cal_prob_topic(self._dict_network_master_dammy, self._list_dict_subnetwork)
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
