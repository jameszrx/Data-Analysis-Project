import datetime, time
import json
import io
import matplotlib.pyplot as plt
import math
import numpy as np
import statsmodels.api as sm
from sklearn import metrics, linear_model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestRegressor

tag = ['tweets_#gohawks.txt', 'tweets_#gopatriots.txt', 'tweets_#nfl.txt',
       'tweets_#patriots.txt', 'tweets_#sb49.txt', 'tweets_#superbowl.txt']

for tagname in tag:
    tweets = []
    num_of_tweets = []
    num_of_retweets = []
    num_of_followers = []
    max_num_of_followers = []
    time_of_day = []
    avg_ranking_score = []
    num_of_inpressions = []
    num_of_momentum = []
    num_of_tweet_favor = []
    num_of_friends = []
    num_of_accelerations = []
    num_of_reply = []
    print()
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
    count_ranking_score = 0
    count_of_momentum = 0
    count_of_inpressions = 0
    count_of_tweet_favor  = 0
    count_of_friends = 0
    count_of_accelerations = 0
    count_of_reply = 0
    
    for tweet in tweets:
        cur_time = tweet['firstpost_date']
        if(cur_time>=(pre_time+3600)):
            num_of_tweets.append(count_of_tweets)
            num_of_retweets.append(count_of_retweets)
            num_of_followers.append(count_of_followers)
            max_num_of_followers.append(count_max_followers)
            time_of_day.append(datetime.datetime.utcfromtimestamp(pre_time).hour)
            if(count_of_tweets==0):
                avg_ranking_score.append(0)
            else:
                avg_ranking_score.append(count_ranking_score/count_of_tweets)
            num_of_momentum.append(count_of_momentum)
            num_of_inpressions.append(count_of_inpressions)
            num_of_tweet_favor.append(count_of_tweet_favor)
            num_of_friends.append(count_of_friends)
            num_of_accelerations.append(count_of_accelerations)
            num_of_reply.append(count_of_reply)
            count_of_tweets = 0
            count_of_retweets = 0
            count_of_followers = 0
            count_max_followers = 0
            count_ranking_score = 0
            count_of_momentum = 0
            count_of_inpressions = 0
            count_of_tweet_favor = 0
            count_of_friends = 0
            count_of_accelerations = 0
            count_of_reply = 0
            pre_time = cur_time
        else:
            count_of_tweets += 1
            #count_of_retweets += tweet['metrics']['citations']['total']
            count_of_retweets += tweet['tweet']['retweet_count']
            count_of_followers += tweet['author']['followers']
            count_max_followers = max(count_max_followers, tweet['author']['followers'])
            count_ranking_score += tweet['metrics']['ranking_score']
            count_of_momentum += tweet['metrics']['momentum']
            count_of_inpressions += tweet['metrics']['impressions']
            count_of_tweet_favor += tweet['tweet']['favorite_count']
            count_of_accelerations += tweet['metrics']['acceleration']
            count_of_reply += tweet['metrics']['citations']['replies']
        

    X_train = np.array([num_of_tweets[:-1], num_of_retweets[:-1],
                        num_of_followers[:-1], max_num_of_followers[:-1],
                        time_of_day[:-1], avg_ranking_score[:-1],
                        num_of_inpressions[:-1], num_of_momentum[:-1],
                        num_of_tweet_favor[:-1], 
                        num_of_accelerations[:-1], num_of_reply[:-1]])
    X_train = X_train.T
    sm.add_constant(X_train)
    Y_train = np.array(num_of_tweets[1:])
    sm_result = sm.OLS(Y_train, X_train).fit()
    Y_predict = sm_result.predict()

    print("rmse for t-test")
    print(math.sqrt(metrics.mean_squared_error(Y_train, Y_predict)))
    print("summary report")
    print(sm_result.summary())

    #Random Forest Regression
    #print("random forest regression socre:")
    #model = RandomForestRegressor(n_estimators=50, max_features=5)
    #reg_score = cross_val_score(model, X_train, Y_train,  cv=10, scoring='mean_squared_error')
    #print(reg_score)

    #scatter plot (change the features you need to plot)
    fig, ax = plt.subplots()
    ax.scatter(X_train[:, 0], Y_predict, color='r', s=60, alpha=0.4)
    plt.title(tagname)
    plt.xlabel("Number of tweets")
    plt.ylabel("Predictant")
    plt.show()

    fig, bx = plt.subplots()
    bx.scatter(X_train[:, 3], Y_predict, color='g', s=60, alpha=0.4)
    plt.title(tagname)
    plt.xlabel("max_num_of_followers")
    plt.ylabel("Predictant")
    plt.show()

    fig, cx = plt.subplots()
    cx.scatter(X_train[:, 9], Y_predict, color='b', s=60, alpha=0.4)
    plt.title(tagname)
    plt.xlabel("num_of accelerations")
    plt.ylabel("Predictant")
    plt.show()


