from django.shortcuts import render, redirect
from .models import contact
from .forms import ContactForm
from django.utils.datastructures import MultiValueDictKeyError
# data processing, CSV file (e.g. pd.read_csv)
import pandas as pd
# linear algebra
import numpy as np
import datetime
from sklearn import svm

# re = regular expression
import re

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# TfidfVectorizer = Transforms text to feature vectors that can be used as input to estimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import metrics

# ML model
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# Create your views here
def home(request):
    if request.method == 'POST':
        input_name = request.POST.get("name1")
        input_email = request.POST.get("email1")
        input_message = request.POST.get("message1")

        contact_info = contact(name=input_name, email=input_email, message=input_message)
        contact_info.save()

    return render(request, 'crop_fertilizer_rainfall_app/home.html', {"send_messages": "Congratulation! your message is Sent successfully"})
    # if request.method == 'POST':
    #     form = ContactForm(request.POST)
    #
    #     if form.is_valid():
    #         instance = form.save(commit=False)
    #         instance.name = request.user
    #         instance.save()
    #         return redirect('/crop_fertilizer_rainfall_predict/home')
    # else:
    #     form = ContactForm()
    #     args = {'form': form}
    #     return render(request, 'crop_fertilizer_rainfall_app/home.html', args)
    # return render(request, 'crop_fertilizer_rainfall_app/home.html', )


# -----------------------------------
#  Crop Recommendation Predict Starts from here
# -----------------------------------
def crop_recommendation_predict(request):
    return render(request, 'crop_fertilizer_rainfall_app/crop_reco.html')


def crop_recommendation_result(request):
    crop_reco_dataset = pd.read_csv(r'crop_fertilizer_rainfall_app/dataset/Crop_recommendation.csv')

    # convert categorical data to numerical values
    crop_reco_dataset.replace({'label': {'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
                                         'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
                                         'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14,
                                         'apple': 15, 'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19,
                                         'jute': 20, 'coffee': 21,
                                         }, }, inplace=True)
    # Label Encoding
    # encoder = LabelEncoder()
    # crop_reco_dataset['label'] = encoder.fit_transform(crop_reco_dataset['label'])

    # Splitting the data set
    X = crop_reco_dataset.drop(['label'], axis=1)
    Y = crop_reco_dataset['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Random Forest
    clf_random_forest=RandomForestClassifier()
    clf_random_forest.fit(X_train, Y_train)

    var1 = float(request.GET['ni'])
    var2 = float(request.GET['pho'])
    var3 = float(request.GET['po'])
    var4 = float(request.GET['temp'])
    var5 = float(request.GET['hum'])
    var6 = float(request.GET['ph'])
    var7 = float(request.GET['rain'])

    pred = clf_random_forest.predict(np.array([var1, var2, var3, var4, var5, var6, var7]).reshape(1, -1))
    print(pred)
    predict = round(pred[0])
    crop = float(predict)
    crop = ""
    if pred == [0]:
        crop = "Rice"
    elif pred == [1]:
        crop = "Maize"
    elif pred == [2]:
        crop = "Chickpea" 
    elif pred == [3]:
        crop = "Kidneybeans"
    elif pred == [4]:
        crop = "Pigeonpeas"
    elif pred == [5]:
        crop = "Mothbeans"
    elif pred == [6]:
        crop = "Mungbean"
    elif pred == [7]:
        crop = "Blackgram"
    elif pred == [8]:
        crop = "Lentil"
    elif pred == [9]:
        crop = "Pomegranate"
    elif pred == [10]:
        crop = "Banana" 
    elif pred == [11]:
        crop = "Mango"
    elif pred == [12]:
        crop = "Grapes"
    elif pred == [13]:
        crop = "Watermelon"
    elif pred == [14]:
        crop = "Muskmelon"
    elif pred == [15]:
        crop = "Apple"
    elif pred == [16]:
        crop = "Orange"   
    elif pred == [17]:
        crop = "Papaya"
    elif pred == [18]:
        crop = "Coconut"
    elif pred == [19]:
        crop = "Cotton"
    elif pred == [20]:
        crop = "Jute"                                                                                                                
    else:
        crop = "Coffee"

    return render(request, 'crop_fertilizer_rainfall_app/crop_reco.html', {'crop_recommendation_predict_result': crop})
# -----------------------------------
#  Crop Recommendation Predict End here
# -----------------------------------


# -----------------------------------
#  RainFall Predict Starts from here
# -----------------------------------
def rainfall_predict(request):
    return render(request, 'crop_fertilizer_rainfall_app/rainfall_prediction.html')

def rainfall_result(request):
    # loading the dataset to pandas DataFrame
    rainfall_dataset = pd.read_csv(r'crop_fertilizer_rainfall_app/dataset/rainfall_prediction.csv')

    # replacing the null values with mean of single column
    # olympic_dataset['Age'].fillna(olympic_dataset.Age.mean(), inplace=True)
    categorical_features = []
    numeric_features = []
    features = rainfall_dataset.columns.values.tolist()
    for col in features:
        if rainfall_dataset[col].dtype != 'object': 
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    for col in numeric_features:
        mean = rainfall_dataset[col].mean()
        rainfall_dataset[col] = rainfall_dataset[col].fillna(mean)
    
    for col in categorical_features:
        mode = rainfall_dataset[col].mode()[0]
        rainfall_dataset[col] = rainfall_dataset[col].fillna(mode)

    # convert categorical values to numerical values
    rainfall_dataset.replace({'WindGustDir': {'NNW': 0, 'NW': 1, 'WNW': 2, 'N': 3, 'W': 4, 'WSW': 5, 'NNE': 6, 'S': 7,
                                              'SSW': 8, 'SW': 9, 'SSE': 10,
                                              'NE': 11, 'SE': 12, 'ESE': 13, 'ENE': 14, 'E': 15}, }, inplace=True)

    rainfall_dataset.replace({'WindDir9am': {'NNW': 0, 'N': 1, 'NW': 2, 'NNE': 3, 'WNW': 4, 'W': 5, 'WSW': 6, 'SW': 7,
                                             'SSW': 8, 'NE': 9, 'S': 10,
                                             'SSE': 11, 'ENE': 12, 'SE': 13, 'ESE': 14, 'E': 15}, }, inplace=True)

    rainfall_dataset.replace({'WindDir3pm': {'NW': 0, 'NNW': 1, 'N': 2, 'WNW': 3, 'W': 4, 'NNE': 5, 'WSW': 6, 'SSW': 7,
                                             'S': 8, 'SW': 9, 'SE': 10,
                                             'NE': 11, 'SSE': 12, 'ENE': 13, 'E': 14, 'ESE': 15}, }, inplace=True)

    rainfall_dataset.replace({'Location': {'Portland': 1, 'Cairns': 2, 'Walpole': 3, 'Dartmoor': 4, 'MountGambier': 5,
                                           'NorfolkIsland': 6,'Albany': 7, 'Witchcliffe': 8, 'CoffsHarbour': 9,
                                           'Sydney': 10,'Darwin': 11, 'MountGinini': 12, 'NorahHead': 13,'Ballarat': 14,
                                           'GoldCoast': 15, 'SydneyAirport': 16, 'Hobart': 17, 'Watsonia': 18,
                                            'Newcastle': 19, 'Wollongong': 20, 'Brisbane': 21, 'Williamtown': 22,
                                           'Launceston': 23, 'Adelaide': 24, 'MelbourneAirport': 25, 'Perth': 26,
                                           'Sale': 27, 'Melbourne': 28, 'Canberra': 29, 'Albury': 30, 'Penrith': 31,
                                            'Nuriootpa': 32, 'BadgerysCreek': 33, 'Tuggeranong': 34, 'PerthAirport': 35,
                                           'Bendigo': 36, 'Richmond': 37, 'WaggaWagga': 38, 'Townsville': 39,
                                           'PearceRAAF': 40, 'SalmonGums': 41, 'Moree': 42, 'Cobar': 43, 'Mildura': 44,
                                            'Katherine': 45, 'AliceSprings': 46, 'Nhil': 47, 'Woomera': 48,
                                           'Uluru': 49}, }, inplace=True)

    # Label Encoding
    encoder = LabelEncoder()
    # convert categorical columns to numerical values
    rainfall_dataset['RainToday'] = encoder.fit_transform(rainfall_dataset['RainToday'])
    rainfall_dataset['RainTomorrow'] = encoder.fit_transform(rainfall_dataset['RainTomorrow'])
    #After label Encoding,  RainToday: NO = 0  YES = 1; RainTomorrow: NO = 0 YES = 1
    print("big problem is start from here")
    rainfall_dataset['Date'] =pd.to_datetime(rainfall_dataset['Date'])
    rainfall_dataset['Date'] = rainfall_dataset['Date'].values.astype(float)

    print("2 big problem is start from here")
    # Splitting the data set
    # X = rainfall_dataset.drop(['Date', 'RainTomorrow'], axis=1)
    X = rainfall_dataset.drop(['RainTomorrow'], axis=1)
    Y = rainfall_dataset['RainTomorrow']

    print("3 big problem is start from here")
    # Splitting the data into Training data & Testing Data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # CatBoostClassifier
    cat = CatBoostClassifier()
    cat.fit(X_train, Y_train)

    # print("Now new problem")
    var1 = (request.GET['date'])
    var1_day = float(pd.to_datetime(var1, format="%Y-%m-%dT").day)
    # var1_month = float(pd.to_datetime(var1, format="%Y-%m-%dT").month)
    var2 = float(request.GET['location'])
    var3 = float(request.GET['mintemp'])
    var4 = float(request.GET['maxtemp'])
    var5 = float(request.GET['rainfall'])
    var6 = float(request.GET['evaporation'])
    var7 = float(request.GET['sunshine'])
    var8 = float(request.GET['windgustdir'])
    var9 = float(request.GET['windgustspeed'])
    var10 = float(request.GET['winddir9am'])
    var11 = float(request.GET['winddir3pm'])
    var12 = float(request.GET['windspeed9am'])
    var13 = float(request.GET['windspeed3pm'])
    var14 = float(request.GET['humidity9am'])
    var15 = float(request.GET['humidity3pm'])
    var16 = float(request.GET['pressure9am'])
    var17 = float(request.GET['pressure3pm'])
    var18 = float(request.GET['cloud9am'])
    var19 = float(request.GET['cloud3pm'])
    var20 = float(request.GET['temp9am'])
    var21 = float(request.GET['temp3pm'])
    var22 = float(request.GET['raintoday'])
    print("all valu are taken")

    pred = cat.predict(np.array([var1_day,  var2, var3, var4, var5, var6, var7, var8, var9,
                                               var10, var11, var12, var13, var14, var15, var16, var17, var18, var19,
                                               var20, var21, var22]).reshape(1, -1))
    
    predict = round(pred[0])
    # rain_tomorrow = (predict)
    # rain_tomorrow = ""
    if pred == [1]:
        # rain_tomorrow = "Yes"
        return render(request, 'crop_fertilizer_rainfall_app/rainy.html')
    else:
        # rain_tomorrow = "No"
        return render(request, 'crop_fertilizer_rainfall_app/sunny.html')

    return render(request, 'crop_fertilizer_rainfall_app/rainfall_prediction.html')
    # return render(request, 'crop_fertilizer_rainfall_app/rainfall_prediction.html', {'rain_tomorrow_result': rain_tomorrow}) 

# -----------------------------------
#  RainFall Predict End here
# -----------------------------------


# ----------------------------------------
#  Fertilizer Prediction Starts from here
# ---------------------------------------
def fertilizer_predict(request):
    return render(request, 'crop_fertilizer_rainfall_app/fertilizer_prediction.html')


def fertilizer_predict_result(request):
    fertilizer_dataset = pd.read_csv(r'crop_fertilizer_rainfall_app/dataset/Fertilizer_Prediction.csv')

    # convert categorical data to numerical values using Label encoding
    # Label Encoding
    encoder = LabelEncoder()
    fertilizer_dataset['Soil Type'] = encoder.fit_transform(fertilizer_dataset['Soil Type'])

    # convert categorical data to numerical values using replace
    fertilizer_dataset.replace({'Crop Type': {'Rice': 0, 'Maize': 1, 'Coffee': 2, 'Sugarcane': 3, 'Cotton': 4,
                                        'Millets': 5, 'Paddy': 6, 'Pulses': 7, 'Wheat': 8, 'Barley': 9, 
                                        'Tobacco': 10, 'Oil seeds': 11, 'Ground Nuts': 12, 'Banana': 13 , 'Mango': 14,
                                        'Apple': 15, 'Orange': 16, 'Coconut': 17, 'Jute': 18, 
                                        },}, inplace=True)

    fertilizer_dataset.replace({'Fertilizer Name': {'Urea': 0, 'DAP': 1, '14-35-14': 2, '28-28': 3, '17-17-17': 4,
                                                '20-20': 5, '10/26/2026': 6, 'Urea_DAP_Potash ': 7, 
                                                'Urea_DAP_Potash': 8, 'NPK_15-5-20+te': 9, '8-10-8_NPK': 10, 
                                                '6-6-6_and_8-3-9-2': 11, 'N-P-K_12-12-12': 12, 'P2O5': 13, 'Granular': 14,
                                                  'N-P-K_12-12-1': 15},}, inplace=True)

    # Splitting the data set
    X = fertilizer_dataset.drop(['Fertilizer Name'], axis=1)
    Y = fertilizer_dataset['Fertilizer Name']

    # Splitting the data into Training data & Testing Data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Random Forest
    # clf_random_forest=RandomForestClassifier()
    # clf_random_forest.fit(X_train, Y_train)
    # log_reg = LogisticRegression()
    # log_reg.fit(X_train, Y_train)

    # CatBoost
    cat = CatBoostClassifier()
    cat.fit(X_train, Y_train)

    var1 = float(request.GET['temp'])
    var2 = float(request.GET['humidity'])
    var3 = float(request.GET['moisture'])
    var4 = float(request.GET['nitrogen'])
    var5 = float(request.GET['potassium'])
    var6 = float(request.GET['phosphorous'])
    var7 = float(request.GET['soiltype'])
    var8 = float(request.GET['croptype'])

    # pred = clf_random_forest.predict(np.array([var1, var2, var3, var4, var5, var6, var7, var8]).reshape(1, -1))
    pred = cat.predict(np.array([var1, var2, var3, var4, var5, var6, var7, var8]).reshape(1, -1))
    print(pred)
    # predict = round(pred[1])
    # fertilizer = float(predict)
    fertilizer = ""
    if pred == [0]:
        fertilizer = "Urea"
    elif pred == [1]:
        fertilizer = "DAP"
    elif pred == [2]:
        fertilizer = "14-35-14" 
    elif pred == [3]:
        fertilizer = "28-28"
    elif pred == [4]:
        fertilizer = "17-17-17"
    elif pred == [5]:
        fertilizer = "20-20"
    elif pred == [6]:
        fertilizer = "10/26/2026"
    elif pred == [7]:
        fertilizer = "Urea_DAP_Potash"
    elif pred == [8]:
        fertilizer = "Urea_DAP_Potash"
    elif pred == [9]:
        fertilizer = "NPK_15-5-20+te"
    elif pred == [10]:
        fertilizer = "8-10-8_NPK" 
    elif pred == [11]:
        fertilizer = "6-6-6_and_8-3-9-2"
    elif pred == [12]:
        fertilizer = "N-P-K_12-12-12"
    elif pred == [13]:
        fertilizer = "P2O5"
    elif pred == [14]:
        fertilizer = "Granular"
    else:
        fertilizer = "N-P-K_12-12-1"    

    return render(request, 'crop_fertilizer_rainfall_app/fertilizer_prediction.html', {'fertilizer_predict_result': fertilizer})
# --------------------------------------
#  Fertilizer Prediction End here
# --------------------------------------

# About Us start
def about_us(request):
    return render(request, 'crop_fertilizer_rainfall_app/about_us.html')
# About Us End
