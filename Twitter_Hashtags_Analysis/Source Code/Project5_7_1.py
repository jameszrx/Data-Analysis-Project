import json
import datetime, time
import matplotlib.pyplot as plt
import math
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_val_score

 
f = open('tweets_#gopatriots.txt', 'r',encoding='UTF-8')
 
line = f.readline()
 
tweets = []
followers = []
retweets = []
while len(line)!=0:
    tweet = json.loads(line)
    tweets.append(tweet)
    line = f.readline()    
 
start_time = tweets[0]['firstpost_date']
end_time = tweets[-1]['firstpost_date']
index_len = math.ceil((end_time - start_time)/3600)
tweet_num = np.zeros(index_len).tolist()
retweet_num = np.zeros(index_len).tolist()
follower_num = np.zeros(index_len).tolist()
max_follower_num = np.zeros(index_len).tolist()
reply_num = np.zeros(index_len).tolist()
mention_num = np.zeros(index_len).tolist()
ranking_scores = np.zeros(index_len).tolist()
favorite_count = np.zeros(index_len).tolist()
impression = np.zeros(index_len).tolist()

for i in range(len(tweets)):
    hour_index = int((tweets[i]['firstpost_date'] - start_time)/3600)
    tweet_num[hour_index] += 1
    retweet_num[hour_index] += tweets[i]['metrics']['citations']['total']
    follower_num[hour_index] += tweets[i]['author']['followers']
    max_follower_num[hour_index] = max(max_follower_num[hour_index], tweets[i]['author']['followers'])  
    reply_num[hour_index] += tweets[i]['metrics']['citations']['replies']
    mention_num[hour_index] += len(tweets[i]['tweet']['entities']['user_mentions'])
    ranking_scores[hour_index] += tweets[i]['metrics']['ranking_score']
    favorite_count[hour_index] += tweets[i]['tweet']['favorite_count']
    impression[hour_index] += tweets[i]['metrics']['impressions']
#find active and burst
active_ones = []
burst_ones = []
for i in range(index_len - 5):
    if tweet_num[i] + tweet_num[i + 1] + tweet_num[i + 2] + tweet_num[i + 3] + tweet_num[i + 4] > 50 :
        active_ones.append(i)
    if len(active_ones) > 0 :
        if tweet_num[i] > max(tweet_num[0] + 50, 1.5*tweet_num[0]):
            burst_ones.append(i)     

active_tweet_num = []
active_retweet_num = []
active_follower_num = []
active_maxfollower_num = []
active_reply_num = []
active_mention_num = []
active_ranking_scores = []
active_favorite_count = []
active_impression = []

burst_tweet_num = []
activeandburst_ones = []
j = 0
for i in active_ones :
    if i == burst_ones[j] - 5:
        activeandburst_ones.append(i)
        active_tweet_num.append(math.log(tweet_num[i]))
        active_retweet_num.append(retweet_num[i])
        active_follower_num.append(follower_num[i])
        active_maxfollower_num.append(max_follower_num[i])
        active_reply_num.append(reply_num[i])
        active_mention_num.append(mention_num[i])
        active_ranking_scores.append(ranking_scores[i])
        active_favorite_count.append(favorite_count[i])
        active_impression.append(impression[i])
        if j < len(burst_ones)-1 : 
            j += 1

print("The number of active time points is:")
print(len(active_ones))
print("The number of bursting time points is:")
print(len(burst_ones))
   
for i in burst_ones :
    burst_tweet_num.append(math.log(tweet_num[i]))
plt.figure()
index = np.arange(index_len).tolist()
width = 1
p = plt.bar(index, tweet_num, width)
plt.xlabel('Time')
plt.ylabel('Number of Tweets per hour')
plt.title('Number of Tweets per hour for #gopatriots')
plt.show()


X_train = np.array([active_tweet_num[:], active_retweet_num[:], active_follower_num[:], active_maxfollower_num[:], active_reply_num[:], active_mention_num[:], active_ranking_scores[:], active_favorite_count[:], active_impression[:], np.ones(len(activeandburst_ones))])
#X_train = np.array([active_tweet_num[:], active_retweet_num[:], active_follower_num[:], active_maxfollower_num[:], np.ones(len(activeandburst_ones))])
X_train = X_train.T
Y_train = np.array(burst_tweet_num[:])

print(sm.OLS(Y_train, X_train).fit().summary())

result=sm.OLS(Y_train, X_train).fit().predict()

#rf=LinearRegression()
#rf.fit(X_train,Y_train)
#predicted = predicted = cross_val_predict(rf, X_train, Y_train, cv=10)
mse = mean_squared_error(result,Y_train)
rmse=math.sqrt(mse)

fig,ax = plt.subplots()
ax.scatter(Y_train, result, s= 60 ,alpha = 0.4)
ax.plot([0,15],[0,15],'k--',lw=2, color = 'r') ## plot line y=x, the range can be changed
ax.set_xlabel('Actual values')
ax.set_ylabel('Fitted values')
plt.show()
#plt.hold(True)
fig,bx=plt.subplots()
residuals=abs(Y_train-result)
bx.scatter(result,residuals, s=60 ,alpha=0.4)
bx.set_xlabel('Fitted values')
bx.set_ylabel('Residuals')
plt.show()

#
#fig2 = plt.subplot()
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
#plt.xlabel('Fitted values')
#plt.ylabel('Residuals')
#plt.show()
