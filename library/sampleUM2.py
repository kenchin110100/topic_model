# coding: utf-8
"""
Unigram Mixtureのサンプル
出展：
'Text Classfication from Labeled and Unlabeled Documents using EM'
K. Nigam, A. K. McCallum and S. Mitchell, Machine Learning, 2000

使い方：
um = UM(alpha=1.0, beta=1.0, K=10, converge=0.01, max_iter=100)
um.set_param(path='path/to/file' or list_d_words=list_list_words)
um.fit()
list_prob = um.infer(list_words)

学習用のファイル:
sample.txt
words in a document in one line sep by single space

元のやつはメモリが足りなくなるので、メモリの省エネを目的に改良
"""

import numpy as np


class UM():
    def __init__(self, alpha=1.0, beta=1.0, K=10, converge=0.01, max_iter=100):
        # 必要なパラメータ
        self.alpha = alpha
        self.beta = beta
        # number of initial K
        self.K = K
        # number of iterations
        self.converge = converge
        # max_iteration
        self.max_iter = max_iter

        self.list_bags = None
        self.list_bags_id = None
        # number of vocaburary
        self.V = None
        # number of documents
        self.D = None
        # likelihood
        self.likelihood = None

        # dictの作成
        self.dict_word_id = None
        self.dict_id_word = None

        # 分布：theta
        self.list_theta = None
        self.list_theta_new = None
        # 分布：phi
        self.list_phi = None
        self.list_phi_new = None
        self.list_dict_phi = None

    # parameterの作成
    def set_param(self, path=None, list_d_words=None):
        # pathセットされたら読み込み、listがセットされたら代入
        if path != None:
            self.list_bags = self.readtxt(path)
        elif list_d_words != None:
            self.list_bags = list_d_words
        else:
            print 'Error: pathかlistをセットしてください'
            return None

        # number of vocaburary
        self.V = len(set([word for row in self.list_bags for word in row]))
        # number of documents
        self.D = len(self.list_bags)

        # dictの作成
        self.dict_word_id, self.dict_id_word = self.make_dict(self.list_bags)
        # bags of word id
        self.list_bags_id = [[self.dict_word_id[word] for word in row]for row in self.list_bags]

    # word_id, id_wordのdict作成
    def make_dict(self, list_bags):
        list_words = [word for row in list_bags for word in row]
        dict_word_id = {word: i for i, word in enumerate(set(list_words))}
        dict_id_word = {i: word for i, word in enumerate(set(list_words))}
        return dict_word_id, dict_id_word

    # テキストファイルの読み込み
    def readtxt(self, path, LF='\n'):
        f = open(path)
        lines = f.readlines()
        f.close
        list_bags = [row.rstrip(LF).split(" ") for row in lines]
        return list_bags

    # 初期化
    def initialize(self):
        self.list_theta = np.random.dirichlet([self.alpha] * self.K)
        self.list_phi = np.random.dirichlet([self.beta] * self.V, self.K)
    
    # 負担率の計算
    def cal_barden_ratio(self, d):
        # オーバーフロー対策
        list_overflow = []
        for z in range(self.K):
            q_dk = 0.0
            q_dk += np.log(self.list_theta[z])
            for v in self.list_bags_id[d]:
                q_dk += np.log(self.list_phi[z][v])
            list_overflow.append(q_dk)
        max_log = np.max(list_overflow)
        list_q = np.array([np.exp(num - max_log) for num in list_overflow])
        return list_q / np.sum(list_q)
    
    # list_phiとlist_thetaを正規化
    def cal_normalization(self):
        self.list_theta /= np.sum(self.list_theta)
        for z in range(self.K):
            self.list_phi[z] /= np.sum(self.list_phi[z])

    # stop判定するためのconvergeの計算
    def cal_likelihood(self):
        list_likelihood = []
        for d in range(self.D):
            # オーバーフロー対策
            list_overflow = []
            l_document = 0.0
            for z in range(self.K):
                # print self.list_theta
                l_document += np.log(self.list_theta[z])
                for v in self.list_bags_id[d]:
                    l_document += np.log(self.list_phi[z][v])
                list_overflow.append(l_document)
            max_log = np.max(list_overflow)
            likelihood = 0.0
            for l_document in list_overflow:
                likelihood += np.exp(l_document - max_log)
            likelihood = np.log(likelihood) + max_log
            list_likelihood.append(likelihood)

        return np.sum(list_likelihood)/len(list_likelihood)


    # メインの学習部分
    def fit(self):
        self.initialize()
        self.likelihood = 0.0
        list_likelihood = []
        for iteration in range(self.max_iter):
            # new_thetaの初期化
            self.list_theta_new = np.array([self.alpha for i in range(self.K)]
                                           , dtype=float)
            # new_phiの初期化
            self.list_phi_new = np.array([[self.beta for j in range(self.V)]
                                          for i in range(self.K)], dtype=float)
            # 負担率を計算
            for d in range(self.D):
                list_q = self.cal_barden_ratio(d)
                self.list_theta_new += list_q
                for word_id in self.list_bags_id[d]:
                    self.list_phi_new[:,word_id] += list_q
            
            # thetaとphiの値を更新
            self.list_theta = self.list_theta_new
            self.list_phi = self.list_phi_new
            self.cal_normalization()
            
            list_likelihood.append(self.cal_likelihood())
            if iteration % 10 == 0:
                print 'finish: ', iteration+1, ' iteration'
                print 'likelihood: ', list_likelihood[-1]
                if len(list_likelihood) > 1:
                    if np.fabs(list_likelihood[-1] - list_likelihood[-2]) < self.converge:
                        break
        self.likelihood = list_likelihood[-1]
        self.list_dict_phi = [{self.dict_id_word[i]: phi
                               for i, phi in enumerate(self.list_phi[z])}
                              for z in range(self.K)]
        print 'finish all: ', self.likelihood

    # クラスの推定
    def infer(self, list_words):
        try:
            # 学習が先に行われていなければエラーを上げる
            if self.list_theta == None:
                raise NameError('calculation first')
            # すべての単語が辞書に含まれていなければエラーを上げる
            for word in list_words:
                if word in self.dict_word_id:
                    break
            else:
                raise KeyError('No word found in dict')
            # オーバーフロー対策
            list_overflow = []
            for i in range(self.K):
                prob = 0.0
                prob += np.log(self.list_theta[i])
                for word in list_words:
                    if word in self.list_dict_phi[i]:
                        prob += np.log(self.list_dict_phi[i][word])
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
