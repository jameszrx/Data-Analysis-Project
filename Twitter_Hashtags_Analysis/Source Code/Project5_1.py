import json
from pprint import pprint
import io
import matplotlib.pyplot as plt

tag = ['tweets_#gohawks.txt', 'tweets_#gopatriots.txt', 'tweets_#nfl.txt',
       'tweets_#patriots.txt', 'tweets_#sb49.txt', 'tweets_#superbowl.txt']

#statistics for each hashtag
for tagname in tag:
    tweets = []
    followers = 0
    retweets = 0
    print(tagname)
    f = io.open(tagname, 'r', encoding = 'utf-8')
    lines = f.readlines()
    for line in lines:
        if(len(line)!=0):
            tweets.append(json.loads(line))

    number_tweets = len(tweets)
    number_hour = (tweets[-1]['firstpost_date']-tweets[0]['firstpost_date'])/3600
    print("number_tweets")
    print(number_tweets)
    print("number of hours")
    print(number_hour)
    print("avg number of tweets")
    print(float(float(number_tweets)/float(number_hour)))

    for tweet in tweets:
        followers += tweet['author']['followers']
        #retweets += tweet['metrics']['citations']['total']
        retweets += tweet['tweet']['retweet_count']

    print("\nnumber_followers")
    print(followers)
    print("avg number of followers")
    print(followers/number_tweets)
    print("\nnumber_retweets")
    print(retweets)
    print("avg number of retweets")
    print(float(float(retweets)/float(number_tweets)))


#plot "number of tweets in hour" over time
count_per_hour = 0
tweets_hour_list = []
tweets = []
pre_time = 0
cur_time = 0
for tagname in tag:
    f = io.open(tagname, 'r', encoding = 'utf-8')
    lines = f.readlines()
    for line in lines:
        if(len(line)!=0):
            tweets.append(json.loads(line))

    for tweet in tweets:
        cur_time = tweet['firstpost_date']
        if(cur_time>=(pre_time+3600)):
            pre_time = cur_time
            tweets_hour_list.append(count_per_hour)
            count_per_hour = 1
        else:
            count_per_hour += 1

    plt.figure(figsize=(25, 10))
    plt.bar(range(len(tweets_hour_list)), tweets_hour_list, color='r')
    plt.title(tagname)
    plt.xlabel('Hours')
    plt.ylabel('Tweets Number')
    plt.show()

