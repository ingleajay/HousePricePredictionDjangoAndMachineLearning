from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from django.http import HttpResponse


def index(request):
    return render(request, 'index.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    data = pd.read_csv(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/js/USA_Housing.csv")
    data = data.drop(['Address'], axis=1)
    X = data.drop('Price', axis=1)
    Y = data['Price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    pred = model.predict(
        np.array([var1, var2, var3, var4, var5]).reshape(1, -1))
    pred = round(pred[0])

    price = "The Predicated price is $ "+str(pred)
    return render(request, 'predict.html', {"result2": price})


def analysis(request):
    data = pd.read_csv(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/js/USA_Housing.csv")
    sns.displot(data=data, x="Price",
                y="Avg. Area Number of Rooms", kind="kde", rug=True)
    plt.savefig(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/img/price_and_room.png")

    sns.displot(data=data, x="Price", y="Avg. Area House Age", kind="kde")
    plt.savefig(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/img/price_and_house.png")

    data = data.drop(['Address'], axis=1)
    X = data.drop('Price', axis=1)
    Y = data['Price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Cofficient'])
    v1 = round(coeff_df.at['Avg. Area Income', 'Cofficient'], 3)
    v2 = round(coeff_df.at['Avg. Area House Age', 'Cofficient'], 3)
    v3 = round(coeff_df.at['Avg. Area Number of Rooms', 'Cofficient'], 3)
    v4 = round(coeff_df.at['Avg. Area Number of Bedrooms', 'Cofficient'], 3)
    v5 = round(coeff_df.at['Area Population', 'Cofficient'], 3)
    prediction = model.predict(X_test)
    error = round(np.sqrt(metrics.mean_absolute_error(Y_test, prediction)), 3)
    return render(request, 'dash.html', {"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "error": error, })


def dataset(request):
    import pandas as pd
    data = pd.read_csv(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/js/USA_Housing_data.csv")
    allData = []
    for i in range(data.shape[0]):
        t = data.iloc[i]
        allData.append(dict(t))
    context = {'dataset': allData}
    return render(request, 'dataset.html', context)


def chart(request):
    data = pd.read_csv(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/js/USA_Housing.csv")

    sns.displot(x="Avg. Area Income", data=data, kde=True)
    plt.savefig(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/img/income.png")

    sns.displot(x="Avg. Area House Age", data=data, kind='kde')
    plt.savefig(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/img/houseage.png")

    sns.displot(x="Avg. Area Number of Rooms",
                data=data, kind='kde')
    plt.savefig(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/img/rooms.png")

    sns.displot(x="Avg. Area Number of Bedrooms",
                data=data)
    plt.savefig(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/img/bedrooms.png")

    sns.displot(x="Area Population", data=data, kind='kde', rug=True)
    plt.savefig(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/img/population.png")

    sns.displot(x="Price", data=data, kind='ecdf')
    plt.savefig(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/img/price.png")

    sns.displot(data=data, x="Price",
                y="Avg. Area Number of Bedrooms", kind="kde")
    plt.savefig(
        r"C:/Users/Ajay's/Desktop/Prediction/HousePricePrediction/ml/static/img/price_and_bedroom.png")
    return render(request, 'chart.html')
