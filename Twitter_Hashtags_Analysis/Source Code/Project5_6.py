import json
import io
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn import metrics


num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
stemmer = PorterStemmer() 

WA = ['wa', 'washington', 'Aberdeen', 'Airway Heights', 'Albion', 'Alderwood',
     'Algona', 'Allyn', 'Almira', 'Amanda Park', 'Amboy', 'American River', 'Anacortes',
     'Anatone', 'Anderson Island', 'Arlington', 'Ashford', 'Asotin', 'Auburn', 'Aukeen',
     'Bainbridge Island', 'Bangor', 'Baring', 'Barstow', 'Battle Ground', 'Belfair',
     'Bellevue', 'Bellingham', 'Benton City', 'Beverly', 'Bingen', 'Birch Bay',
     'Black Diamond', 'Blaine', 'Bonney Lake', 'Bothell', 'Bow', 'Bremerton', 'Brewster',
     'Bridgeport', 'Brier', 'Brinnon', 'Brownsville', 'Buckley', 'Bucoda', 'Burien',
     'Burley', 'Burlington', 'Camano Island', 'Camas', 'Canyon Park', 'Carbonado',
     'Carnation', 'Carson', 'Cascade', 'Cashmere', 'Castle Rock', 'Cathan', 'Cathart',
     'Cathlamet', 'Central Park', 'Centralia', 'Chehalis', 'Chelan', 'Cheney', 'Chewelah',
     'Chico', 'Chinook', 'Clarkston', 'Cle Elum', 'Clinton', 'Clyde Hill', 'Coles Corner',
     'Colfax', 'College Place', 'Colton', 'Colville', 'Concrete', 'Conconully',
     'Cosmopolis', 'Cougar', 'Coulee City', 'Coulee Dam', 'Coupeville', 'Crescent Bar',
     'Crystal Mountain', 'Cusick',  'Darrington', 'Davenport',
     'Dayton', 'Deer Park', 'Deming', 'Des Moines', 'Diablo', 'Discovery Bay', 'Dishman',
     'Dupont', 'Duvall', 'Eatonville', 'Easton', 'Echo Lake', 'Edgewood', 'Edmonds',
     'Elbe', 'Eldon', 'Electric City', 'Ellensburg', 'Elma', 'Elmer City', 'Enumclaw',
     'Ephrata', 'Evans', 'Everett', 'Federal Way', 'Ferndale', 'Fife', 'Firecrest',
     'Forks', 'Fortson', 'Fox Island', 'Freeland', 'Friday Harbor', 'Garfield', 'George',
     'Gig Harbor', 'Glacier', 'Glenoma', 'Glenwood', 'Goldbar', 'Goldendale', 'Gorst',
     'Graham', 'Grand Coulee', 'Grandview', 'Granger', 'Granite Falls', 'Grandview', 
     'Grapeview', 'Greenbank', 'Greenwater', 'Guemes Island', 'Hansville', 'Harrah',
     'Harrington', 'Hazel', 'Holly', 'Hood Canal', 'Hoodsport', 'Hoquiam', 'Humptulips',
     'Husum', 'Ilwaco', 'Index', 'Indianola', 'Ione', 'Issaquah', 'Kalaloch', 'Kalama',
     'Kelso', 'Kennewick', 'Kent', 'Kettle Falls', 'Key Center', 'Keyport', 'Kingston',
     'Kirkland', 'Kittitas', 'Klickitat', 'La Conner', 'La Push', 'Lacey', 'Lake Stevens',
     'Lakewood', 'Langley', 'Leavenworth', 'Liberty Lake', 'Lilliwaup', 'Little Boston', 
     'Long Beach', 'Longview', 'Lummi Island', 'Lyle', 'Lynden', 'Lynnwood', 'Machias',
     'Maltby', 'Manchester', 'Manson', 'Maple Falls', 'Maple Valley', 'Marblemount',
     'Marysville', 'Mattawa', 'McCleary', 'Mill Creek', 'Moxee City', 'Mazama', 
     'Medical Lake', 'Medina', 'Mercer Island', 'Metaline Falls', 'Methow', 'Mill Creek',
     'Milton', 'Mineral', 'Moclips', 'Monroe', 'Montesano', 'Morton', 'Moses Lake',
     'Mossyrock', 'Mount Adams', 'Mount Baker', 'Mount Vernon', 'Mountlake Terrace',
     'Moxee City', 'Mukilteo', 'Naches', 'Nahcotta', 'Naselle', 'Navy Yard City',
     'Neah Bay', 'Nespelem', 'Newhalem', 'Newport', 'Nile', 'Nooksack', 'North Bend',
     'North Bonneville', 'Northwoods', 'Oak Harbor', 'Oakville', 'Ocean City',
     'Ocean Park', 'Ocean Shores', 'Odessa', 'Okanogan', 'Olalla', 'Olympia', 
     'Omak', 'Opportunity', 'Orient', 'Oroville', 'Orting', 'Oso', 'Othello', 
     'Outlook', 'Ozette', 'Pacific Beach', 'Packwood', 'Pasco', 'Pateros', 'Paterson', 
     'Pe Ell', 'Peshastin', 'Plain', 'Pomeroy', 'Port Angeles', 'Port Gamble',
     'Port Hadlock', 'Port Ludlow', 'Port Orchard', 'Port Townsend', 'Potlatch',
     'Poulsbo', 'Prescott', 'Prosser', 'Pullman', 'Purdy', 'Puyallup', 'Quilcene', 
     'Quincy', 'Quinault', 'Randle', 'Raymond', 'Reardon', 'Redmond', 'Renton', 
     'Republic', 'Richland', 'Rimrock', 'Ritzville', 'Riverside', 'Robe', 'Rock Island', 
     'Rockport', 'Rosalia', 'Roslyn', 'Royal City', 'Salkum', 'Sammamish', 'Seabeck', 
     'SeaTac', 'Seattle', 'Seaview', 'Sedro Woolley', 'Selah', 'Sequim', 'Shelton',
     'Shoreline', 'Silvana', 'Silverdale', 'Silverlake', 'Skamokawa', 'Skykomish', 
     'Smokey Point', 'Snohomish', 'Snoqualmie', 'Snoqualmie Pass', 'Soap Lake',
     'South Bend', 'South Colby', 'Southworth', 'Spanaway', 'Spokane', 'Spokane Valley',
     'Sprague', 'Springdale', 'Stanwood', 'Startup', 'Stehekin', 'Steilacoom', 
     'Stevens Pass', 'Stevenson', 'Sultan', 'Sumas', 'Sumner', 'Suquamish', 'Sultan',
     'Sunnyside', 'Suquamish', 'Tacoma', 'Tahuya', 'Tekoa', 'Tenino', 'Thorp', 
     'Tokeland', 'Tonasket', 'Toppenish', 'Toutle', 'Tracyton', 'Trout Lake', 'Tukwila',
     'Tulalip', 'Tumwater', 'Twisp', 'Union', 'Union Gap', 'Uniontown', 'Usk', 
     'Van Horn', 'Vancouver', 'Vantage', 'Vashon Island', 'Verlot', 'Waitsburg',
     'Walla Walla', 'Wallula', 'Wapato', 'Warden', 'Washougal', 'Washtucna', 'Waterville',
     'Wauconda', 'Wellpinit', 'Wenatchee', 'West Richland', 'Westport', 'White Salmon',
     'Wilbur', 'Winlock', 'Winthrop', 'Wishram', 'Woodinville', 'Woodland', 'Woodway',
     'Yacolt', 'Yakima', 'Yelm', 'Zillah']
key_WA = []
for word in WA:
    key_WA.append(word.lower())

tweets = []
text = []
actual_location = []


f = io.open('./tweet_data/tweets_#superbowl.txt', 'r', encoding = 'utf-8')
#lines = f.readlines()
for i in range (1000000):
    line = f.readline()
#for line in lines:
    if(len(line)!=0):
        tweets.append(json.loads(line))

for tweet in tweets:
        location = tweet['tweet']['user']['location'].lower()
        if 'washington' in location and not 'dc' in location:
            actual_location.append(0) # 0 for 'WA'
            tweet_text = tweet['tweet']['text'].lower()
            tweet_text = [w for w in tweet_text.split() if not w in stop_words ]
            tweet_text = " ".join([stemmer.stem(plural) for plural in tweet_text])
            for b in num:
                tweet_text = tweet_text.replace(b,"")
            for c in string.punctuation:
                tweet_text = tweet_text.replace(c,"")
            text.append(tweet_text)
            print tweet_text
            print '*********'

        elif 'massachusetts' in location:
            actual_location.append(1) # 1 for 'MA'
            tweet_text = tweet['tweet']['text'].lower()
            tweet_text = [w for w in tweet_text.split() if not w in stop_words ]
            tweet_text = " ".join([stemmer.stem(plural) for plural in tweet_text])
            for b in num:
                tweet_text = tweet_text.replace(b,"")
            for c in string.punctuation:
                tweet_text = tweet_text.replace(c,"")
            text.append(tweet_text)
            print tweet_text
            print '*********'

count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(text)
transformer = TfidfTransformer(use_idf=True).fit(train_counts)
train_tfidf = transformer.transform(train_counts)

svd = TruncatedSVD(n_components=50, random_state=42)
feature_train = svd.fit_transform(train_tfidf)
feature_train= Normalizer(copy=False).fit_transform(feature_train)

print 'GaussianNB'                            
clf = GaussianNB()
clf.fit(feature_train,actual_location)
predict_location = clf.predict(feature_train)
accuracy = np.mean(predict_location == actual_location)
print 'The accuracy for the model is %f' % accuracy
print "The precision and recall values are:"
print metrics.classification_report(actual_location, predict_location)
print 'The confusion matrix is as shown below:'
print metrics.confusion_matrix(actual_location, predict_location)
probas_ = clf.predict_proba(feature_train)                                    
fpr, tpr, thresholds = roc_curve(actual_location, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "Naive Bayes")

print 'SVM'
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(feature_train,actual_location)
predict_location = clf.predict(feature_train)
accuracy = np.mean(predict_location == actual_location)
print 'The accuracy for the model is %f' % accuracy
print "The precision and recall values are:"
print metrics.classification_report(actual_location, predict_location)
print 'The confusion matrix is as shown below:'
print metrics.confusion_matrix(actual_location, predict_location)
probas_ = clf.predict_proba(feature_train)                                    
fpr, tpr, thresholds = roc_curve(actual_location, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "SVM")

print 'LogisticRegression'
clf = LogisticRegression()
clf.fit(feature_train,actual_location)
predict_location = clf.predict(feature_train)
accuracy = np.mean(predict_location == actual_location)
print 'The accuracy for the model is %f' % accuracy
print "The precision and recall values are:"
print metrics.classification_report(actual_location, predict_location)
print 'The confusion matrix is as shown below:'
print metrics.confusion_matrix(actual_location, predict_location)
probas_ = clf.predict_proba(feature_train)                                    
fpr, tpr, thresholds = roc_curve(actual_location, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "Logistic Regression")

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.legend(loc="lower right")

plt.show()
        








