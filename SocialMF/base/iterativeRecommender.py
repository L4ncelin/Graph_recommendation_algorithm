from base.recommender import Recommender
from util import config
import numpy as np
from random import shuffle
from util.measure import Measure
from util.qmath import find_k_largest
import matplotlib.pyplot as plt

class IterativeRecommender(Recommender):
    def __init__(self, conf, trainingSet, testSet, fold='[1]'):
        super(IterativeRecommender, self).__init__(conf, trainingSet, testSet, fold)
        self.bestPerformance = []
        self.earlyStop = 0

        # Add these for tracking
        self.loss_history = []
        self.metric1_history = []
        self.metric2_history = []

    def readConfiguration(self):
        super(IterativeRecommender, self).readConfiguration()
        self.emb_size = int(self.config['num.factors'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        learningRate = config.OptionConf(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        if self.evalSettings.contains('-tf'):
            self.batch_size = int(self.config['batch_size'])
        regular = config.OptionConf(self.config['reg.lambda'])
        self.regU, self.regI, self.regB = float(regular['-u']), float(regular['-i']), float(regular['-b'])

    def printAlgorConfig(self):
        super(IterativeRecommender, self).printAlgorConfig()
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Regularization parameter: regU %.3f, regI %.3f, regB %.3f' % (self.regU, self.regI, self.regB))
        print('=' * 80)

    def initModel(self):
        self.P = np.random.rand(len(self.data.user), self.emb_size) / 3
        self.Q = np.random.rand(len(self.data.item), self.emb_size) / 3
        self.loss, self.lastLoss = 0, 0

    def trainModel_tf(self):
        import tensorflow as tf
        self.u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.r = tf.placeholder(tf.float32, [None], name="rating")
        self.U = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.005), name='U')
        self.V = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.005), name='V')
        self.user_biases = tf.Variable(tf.truncated_normal(shape=[self.num_users, 1], stddev=0.005), name='U')
        self.item_biases = tf.Variable(tf.truncated_normal(shape=[self.num_items, 1], stddev=0.005), name='U')
        self.user_bias = tf.nn.embedding_lookup(self.user_biases, self.u_idx)
        self.item_bias = tf.nn.embedding_lookup(self.item_biases, self.v_idx)
        self.user_embedding = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.item_embedding = tf.nn.embedding_lookup(self.V, self.v_idx)

    def updateLearningRate(self, epoch):
        if epoch > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.lRate *= 1.05
            else:
                self.lRate *= 0.5
        if self.lRate > self.maxLRate > 0:
            self.lRate = self.maxLRate

    def predictForRating(self, u, i):
        if self.data.containsUser(u) and self.data.containsItem(i):
            return self.P[self.data.user[u]].dot(self.Q[self.data.item[i]])
        elif self.data.containsUser(u) and not self.data.containsItem(i):
            return self.data.userMeans[u]
        elif not self.data.containsUser(u) and self.data.containsItem(i):
            return self.data.itemMeans[i]
        else:
            return self.data.globalMean

    def predictForRanking(self, u):
        if self.data.containsUser(u):
            return self.Q.dot(self.P[self.data.user[u]])
        else:
            return [self.data.globalMean] * self.num_items

    def isConverged(self, epoch):
        from math import isnan
        if isnan(self.loss):
            print('Loss = NaN or Infinity: current settings do not fit the recommender! Change the settings and try again!')
            exit(-1)
        deltaLoss = self.lastLoss - self.loss

        if self.ranking.isMainOn():
            print('%s %s epoch %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f' \
                  % (self.modelName, self.foldInfo, epoch, self.loss, deltaLoss, self.lRate))
        else:
            measure = self.rating_performance()
            print('%s %s epoch %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %5s %5s' \
                  % (self.modelName, self.foldInfo, epoch, self.loss, deltaLoss, self.lRate,
                     measure[0].strip()[:11], measure[1].strip()[:12]))

            # Save loss and metrics for plotting
            self.loss_history.append(self.loss)
            try:
                self.metric1_history.append(float(measure[0].strip().split(':')[-1]))
                self.metric2_history.append(float(measure[1].strip().split(':')[-1]))
            except ValueError:
                self.metric1_history.append(0)
                self.metric2_history.append(0)

        # Check convergence
        converged = abs(deltaLoss) < 1e-3
        if not converged:
            self.updateLearningRate(epoch)

        self.lastLoss = self.loss
        shuffle(self.data.trainingData)

        if epoch == 15:
            import matplotlib.pyplot as plt

            train_loss = self.loss_history
            rmse = self.metric1_history      
            mae = self.metric2_history       

            epochs = range(1, len(train_loss) + 1)

            plt.figure(figsize=(14, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_loss, 'bo-', label='Train Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Evolution of Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            l1 = ax1.plot(epochs, rmse, 'deepskyblue', marker='o', label='RMSE')
            l2 = ax2.plot(epochs, mae, 'indianred', marker='o', label='MAE')

            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('RMSE', color='deepskyblue')
            ax2.set_ylabel('MAE', color='indianred')
            plt.title('Evolution of RMSE and MAE per Epoch')

            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')

            plt.tight_layout()

            plt.savefig('plots/training_metrics.png', dpi=300)
            plt.show()


        return converged

    def rating_performance(self):
        res = []
        for ind, entry in enumerate(self.data.testData):
            user, item, rating = entry
            prediction = self.predictForRating(user, item)
            pred = self.checkRatingBoundary(prediction)
            res.append([user, item, rating, pred])
        self.measure = Measure.ratingMeasure(res)
        return self.measure

    def ranking_performance(self, epoch):
        top = self.ranking['-topN'].split(',')
        top = [int(num) for num in top]
        N = max(top)
        recList = {}
        print('Evaluating...')
        for user in self.data.testSet_u:
            candidates = self.predictForRanking(user)
            ratedList, ratingList = self.data.userRated(user)
            for item in ratedList:
                candidates[self.data.item[item]] = 0
            ids, scores = find_k_largest(N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            recList[user] = list(zip(item_names, scores))
        measure = Measure.rankingMeasure(self.data.testSet_u, recList, [N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.saveModel()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
                self.bestPerformance.append(performance)
            self.saveModel()
        print('-' * 120)
        print('Quick Ranking Performance ' + self.foldInfo + ' (Top-' + str(N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join(measure))
        bp = 'Precision:' + str(self.bestPerformance[1]['Precision']) + ' | '
        bp += 'Recall:' + str(self.bestPerformance[1]['Recall']) + ' | '
        bp += 'F1:' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'MDCG:' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure
