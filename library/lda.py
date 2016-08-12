# coding: utf-8
"""
LDAをCGSで計算するためのコード
"""

import numpy
from scipy.special import digamma
from scipy.spatial.distance import cosine

class LDA:
    def __init__(self, K, alpha, beta, docs, V):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs
        self.V = V
        self.D = len(self.docs)
        self.N_d = numpy.array([len(row) for row in self.docs])

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((len(self.docs), K))     # word count of each document and topic
        self.n_z_t = numpy.zeros((K, V)) # word count of each topic and vocabulary
        self.n_z = numpy.zeros(K)    # word count of each topic

        self.N = 0
        for m, doc in enumerate(docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                z = numpy.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(numpy.array(z_n))

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = (self.n_z_t[:, t]+self.alpha) * (n_m_z+self.beta) / (self.n_z+self.V*self.beta)
                try:
                    new_z = numpy.random.choice(self.K, 1, p= p_z / p_z.sum())[0]
                except ValueError:
                    print p_z / p_z.sum()
                    raise

                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1
                
    def cal_param(self):
        """
        ハイパーパラメータの更新
        """
        alpha_1 = numpy.sum(digamma(self.n_m_z+self.alpha))-self.D*self.K*digamma(self.alpha)
        alpha_2 = self.K * numpy.sum(digamma(self.N_d+self.alpha*self.K)) - self.D*self.K*digamma(self.alpha*self.K)
        beta_1 = numpy.sum(digamma(self.n_z_t+self.beta)) - self.K*self.V*digamma(self.beta)
        beta_2 = self.V*numpy.sum(self.n_z+self.beta*self.V) - self.K*self.V*digamma(self.beta*self.V)

        self.alpha *= alpha_1/alpha_2
        self.beta *= beta_1/beta_2

    def worddist(self):
        """get topic-word distribution"""
        return (self.n_z_t+self.beta) / (self.n_z[:, numpy.newaxis]+self.beta*self.V)

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = (self.n_m_z[m]+self.alpha) / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return log_per / N
    
    def cal_topic(self):
        list_theta = []
        for m, doc in enumerate(self.docs):
            theta = (self.n_m_z[m]+self.alpha) / (len(self.docs[m]) + self.K*self.alpha)
            list_theta.append(theta)
        return list_theta
    
    def cal_ave_dis(self):
        list_phi = self.worddist()
        list_dis = [1-cosine(list_phi[i], list_phi[j]) for i in range(self.K) for j in range(i+1, self.K)]
        return numpy.sum(list_dis) / len(list_dis)

def lda_learning(lda, iteration, voca, converge=0.01):
    pre_perp = lda.perplexity()
    #print ("initial perplexity=%f" % pre_perp)
    for i in range(iteration):
        lda.inference()
        lda.cal_param()
        if i % 10 == 0:
            perp = lda.perplexity()
            #print ("-%d p=%f" % (i + 1, perp))
            if pre_perp-perp < converge:
                break
            pre_perp = perp
    #output_word_topic_dist(lda, voca)

def output_word_topic_dist(lda, voca):
    zcount = numpy.zeros(lda.K, dtype=int)
    wordcount = [dict() for k in range(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()
    for k in range(lda.K):
        print ("\n-- topic: %d (%d words)" % (k, zcount[k]))
        for w in numpy.argsort(-phi[k])[:20]:
            print ("%s: %f (%d)" % (voca[w], phi[k,w], wordcount[k].get(w,0)))