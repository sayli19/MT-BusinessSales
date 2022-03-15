import pandas as pd
from flask import Flask, render_template

from Dashboard import fetchCommonWords, fetchEmotionPer, fetchStarPercentage
from LandingPage import highRatings, lowRatings, neutralRatings, productCSV
from Recommendation import get_recommendations

app = Flask(__name__)

subpolDF = pd.read_csv(
    'E:/Master-Thesis/OptimizingBusinessSales/dataset/SubPol.csv')


@app.route("/")
def hello_world():
    sayli = 11
    return render_template('Demo.html', sayli=sayli)


@ app.route("/landing-page")
def landingPage():

    NeutralList = list(neutralRatings.values)
    PositiveList = list(highRatings.values)
    NegativeList = list(lowRatings.values)
    return render_template('LandingPage.html', neutralList=NeutralList, positiveList=PositiveList, negativeList=NegativeList)


@ app.route('/dashboard/<item>', methods=['GET'])
def dashboard(item):
    getObjectByID = productCSV.loc[productCSV['ID'] == int(item)]
    itemID = list(getObjectByID.ID)
    itemName = list(getObjectByID.name)
    itemDescription = list(getObjectByID.Description)
    itemImg = list(getObjectByID.imageURLs)
    itemPrice = list(getObjectByID.Price)
    starPercentage = fetchStarPercentage(itemID)

    # sentimental analysis
    arrCommon = fetchCommonWords(itemID, itemName)

    aspectsOne = arrCommon
    # print(list(arrCommon.values))

    # polarity and sujectivity
    arrSPDF = subpolDF.loc[subpolDF['ID'] == int(item)]
    arrSUB = arrSPDF['Subjectivity']
    avgSUB = sum(arrSUB)/len(arrSUB)
    roundOfSUB = round(avgSUB, 2)

    arrAnalysis = arrSPDF['Analysis']
    posCount = 0
    negCount = 0
    neuCount = 0

    for item in arrAnalysis:
        if item == "Positive":
            posCount += 1
        elif item == "Negative":
            negCount += 1
        else:
            neuCount += 1

    arrEmotions = fetchEmotionPer(posCount, negCount, neuCount)

    return render_template('Dashboard.html', title='Details', ID=item, itemName=itemName, itemDescription=itemDescription, itemImg=itemImg, itemPrice=itemPrice, starPercentage=starPercentage, arrCommon=arrCommon, itemID=itemID, roundOfSUB=roundOfSUB, posCount=posCount, negCount=negCount, neuCount=neuCount, arrEmotions=arrEmotions)


@ app.route('/recommendation/<item>', methods=['GET'])
def recommendation(item):
    getObjectByID = productCSV.loc[productCSV['ID'] == int(item)]

    itemID = list(getObjectByID.ID)
    itemName = list(getObjectByID.name)
    itemDescription = list(getObjectByID.Description)
    itemImg = list(getObjectByID.imageURLs)
    itemPrice = list(getObjectByID.Price)
    arrOfRec = get_recommendations(itemName[0])
    arrOfRec = (arrOfRec.values)
    return render_template('Recommendation.html', arrOfRec=arrOfRec,  itemName=itemName, itemDescription=itemDescription, itemImg=itemImg, itemPrice=itemPrice, ogID=itemID)


@ app.route('/compare/<OGID>/<CID>', methods=['GET'])
def comparePage(OGID, CID):

    # first object
    OGObj = productCSV.loc[productCSV['ID'] == int(OGID)]
    ogID = list(OGObj.ID)
    ogName = list(OGObj.name)
    ogImg = list(OGObj.imageURLs)
    ogPrice = list(OGObj.Price)
    ogRating = list(OGObj.Avg)
    ogStarPer = fetchStarPercentage(ogID)

    # first object  - sentimental analysis
    ogCommonWords = fetchCommonWords(ogID, ogName)

    # Compared object
    comparedObj = productCSV.loc[productCSV['ID'] == int(CID)]
    cID = list(comparedObj.ID)
    cName = list(comparedObj.name)
    cImg = list(comparedObj.imageURLs)
    cPrice = list(comparedObj.Price)
    cRating = list(comparedObj.Avg)
    cStarPer = fetchStarPercentage(cID)

    # first object  - sentimental analysis
    cCommonWords = fetchCommonWords(cID, cName)

    return render_template('Compare.html', ogName=ogName, ogImg=ogImg, ogRating=ogRating, cName=cName, cImg=cImg, cRating=cRating, ogStarPer=ogStarPer, cStarPer=cStarPer, ogCommonWords=ogCommonWords, cCommonWords=cCommonWords)


if __name__ == "__main__":
    app.run(debug=True)
