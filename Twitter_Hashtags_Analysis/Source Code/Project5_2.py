import datetime, time
import json
import io
import matplotlib.pyplot as plt
import math
import numpy as np
import statsmodels.api as sm
from sklearn import metrics

tag = ['tweets_#gohawks.txt', 'tweets_#gopatriots.txt', 'tweets_#nfl.txt',
       'tweets_#patriots.txt', 'tweets_#sb49.txt', 'tweets_#superbowl.txt']

for tagname in tag:
    tweets = []
    num_of_tweets = []
    num_of_retweets = []
    num_of_followers = []
    max_num_of_followers = []
    time_of_day = []
    print(tagname)
    f = io.open(tagname, 'r', encoding = 'utf-8')
    lines = f.readlines()
    for line in lines:
        if(len(line)!=0):
            tweets.append(json.loads(line))

    pre_time = 0
    cur_time = 0
    count_of_tweets = 0
    count_of_retweets = 0
    count_of_followers = 0
    count_max_followers = 0
    for tweet in tweets:
        cur_time = tweet['firstpost_date']
        if(cur_time>=(pre_time+3600)):
            num_of_tweets.append(count_of_tweets)
            num_of_retweets.append(count_of_retweets)
            num_of_followers.append(count_of_followers)
            max_num_of_followers.append(count_max_followers)
            time_of_day.append(datetime.datetime.utcfromtimestamp(pre_time).hour)
            pre_time = cur_time
            count_of_tweets = 0
            count_of_retweets = 0
            count_of_followers = 0
            count_max_followers = 0
        else:
            count_of_tweets += 1
            count_of_retweets += tweet['metrics']['citations']['total']
            #count_of_retweets += tweet['tweet']['retweet_count']
            count_of_followers += tweet['author']['followers']
            count_max_followers = max(count_max_followers, tweet['author']['followers'])

    X_train = np.array([num_of_tweets[:-1], num_of_retweets[:-1], num_of_followers[:-1], max_num_of_followers[:-1], time_of_day[:-1]])
    X_train = X_train.T
    sm.add_constant(X_train)
    Y_train = np.array(num_of_tweets[1:])
    sm_result = sm.OLS(Y_train, X_train).fit()
    Y_predict = sm_result.predict()

    print("rmse")
    print(math.sqrt(metrics.mean_squared_error(Y_train, Y_predict)))
    print("summary report")
    print(sm_result.summary())
