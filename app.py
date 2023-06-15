'''
Yousif Alnajjar - 112 TP - UI
Sources:
https://github.com/Manik9/LSTMs
'''
import numpy as np

from cmu_graphics import *
from PIL import Image
import yfinance as yf
import math, copy
from datetime import datetime
import LSTM


def startBalance():
    return 10000

def onAppStart(app):
    app.height = 720
    app.width = 1280
    app.steps = 1
    app.stepsPerSecond = 3
    app.modelLoaded = False
    app.dataLoaded = False
    app.backgroundImg = CMUImage(Image.open('background.jpg')) # created on https://www.photopea.com/
    app.logoImg= CMUImage(Image.open('logo.png')) # https://openclipart.org/detail/256789/rocket edited on photopea.com
    app.screen = 'splash'
    app.dataLen = 10
    app.pointsAnalysed = 1
    app.boxWidth, app.boxHeight = 500, 600
    app.boxLeft = (app.width // 2) - (app.boxWidth // 2)
    app.boxTop = (app.height // 2) - (app.boxHeight // 2)
    app.riskLevels = ['Low Risk', 'Medium Risk', 'High Risk']
    app.clicked = False
    app.rowLabels = []
    app.colLabels = []
    app.waitSteps = 0
    app.graphHeight = app.height - 150
    app.graphWidth = (app.width * (4 / 5)) - 150
    app.selectionActive = False
    app.newDataFound = True
    app.split = pythonRound(app.width * (4 / 5))

    #model defaults
    np.random.seed(15112)
    app.iter = 0
    app.prev = 20
    app.forward = 5
    app.cells = 100
    app.params = LSTM.initParams(app.cells, app.prev)
    app.nodes = []


def redrawAll(app):
    if app.screen == 'splash': drawSplash(app)
    if app.screen == 'model-train': drawTraining(app)
    if app.screen == 'risk-select': drawSelect(app)
    if app.screen == 'main': drawMain(app)


def drawMain(app):
    drawLine(pythonRound(app.width * (4/5)), 0, pythonRound(app.width * (4/5)), app.height)
    drawLine(pythonRound(app.width * (4/5)), 100, app.width, 100)
    drawLine(pythonRound(app.width * (4/5)), 140, app.width, 140)
    drawLine(pythonRound(app.width * (4/5)), 180, app.width, 180)
    drawGraph(app)
    drawOrderbook(app)


def drawOrderbook(app):
    drawBalance(app)
    drawLabel(f'Current price: {app.displayedVals[app.forward]:0.2f}',
              app.split + ((app.width - app.split) // 2), 120)
    drawLabel(f'{app.forward * 3}-min price prediction: {app.displayedVals[0]:0.2f}',
              app.split + ((app.width - app.split) // 2), 160)
    drawOrders(app)


def drawOrders(app):
    drawOpenOrders(app)
    drawClosedOrder(app)


def drawClosedOrder(app):
    drawLine(app.split, 600, app.width, 600)
    if len(app.closedTrades) == 0:
        drawLabel('No closed orders yet!', app.split + ((app.width - app.split) // 2), 660, fill='grey')
    else:
        buy, sell, amount, time = app.closedTrades[-1]
        drawLabel(f'{amount:0.2f} BTC @ {buy:0.2f}', app.split + ((app.width - app.split) // 2), 645, size=20)
        drawLabel(f'Sold @ {sell:0.2f}', app.split + ((app.width - app.split) // 2), 675, size=20)


def drawOpenOrders(app):
    if len(app.openTrades) == 0:
        drawLabel('No open orders!', app.split + ((app.width - app.split) // 2), 390, fill='grey')
    else:
        top = 180
        boxHeight = 80
        displayedTrades = range(len(app.openTrades[-5:]))
        for i in displayedTrades:
            price, amount, time = app.openTrades[displayedTrades[-i]]
            drawLabel(f'{amount:0.2f}', app.split + 20,
                      (top + 30) + (boxHeight * i), size=40, align='left')
            drawLabel('BTC', app.split + 60, (top + 60) + (boxHeight * i), size=10)
            drawLine(app.split, top+((i+1)*boxHeight), app.width, top+((i+1)*boxHeight))
            fill = 'green' if price < app.displayedVals[app.forward] else 'red'
            drawLabel(f'@ {price:0.02f}', app.split + ((app.width - app.split) // 2),
                      (top + 25) + (boxHeight * i), size=20, align='left', fill=fill)
            drawLabel(f'{time}', app.width - 20, (top + 55) + (boxHeight * i), size=20, align='right')


def drawBalance(app):
    totalBal = app.balance + (app.balanceBTC * app.displayedVals[app.forward])
    drawLabel('Current Balance:', app.split + ((app.width - app.split) //2), 30, size=30)
    drawLabel(f'${totalBal:0.2f}', app.split + ((app.width - app.split) // 2), 60, size=20)
    if totalBal > startBalance(): color='green'
    elif totalBal < startBalance(): color='red'
    else: color='grey'
    drawLabel(f'{totalBal - startBalance():0.2f}', app.split + ((app.width - app.split) // 2), 80, size=16, fill=color)


def drawGraph(app):
    drawLabel('Time', app.graphWidth // 2 + 100, app.height - 30)
    drawLabel('Price (USD)', 30, app.graphHeight // 2, rotateAngle=270)
    drawGridAndYLabels(app)
    drawXLabels(app)
    drawLines(app)


def drawLines(app):
    for i in range(len(app.yPlots) - 1):
        dashes = True if i >= len(app.yPlots) - 6 else False
        drawLine(100 + (i*app.colWidth), app.yPlots[i], 100 + ((i+1)*app.colWidth), app.yPlots[i+1], fill='red',
                 dashes=dashes)


def drawGridAndYLabels(app):
    for i in range(app.graphRows):
        y = 50 + (i * app.rowHeight)
        drawLine(100, y, 100 + app.graphWidth, y)
        drawLabel(app.rowLabels[-(i+1)], 90, y, align='right')


def drawXLabels(app):
    for i in range(app.graphCols):
        drawLabel(app.colLabels[i], 100 + (i*(app.colWidth)), app.height - 70)


def drawSelect(app):
    drawImage(app.backgroundImg, 0, 0)
    drawRectWithShadow(app, app.width // 2 - app.boxWidth // 2, app.height // 2 - app.boxHeight // 2,
                       app.boxWidth, app.boxHeight)
    drawLabel('Select your risk level:', app.width // 2, 100, size = 20)
    for i in range(3):
        fill = 'black' if app.selected == i else None
        drawLabel(app.riskLevels[i], app.boxLeft + 120,
                  app.boxTop + (((app.boxHeight - 150) // 3) * i) + ((app.boxHeight - 150) // 6) + 40,
                  size=40, align='left')
        drawRect(app.boxLeft + 50, app.boxTop + (((app.boxHeight - 150) // 3) * i) + ((app.boxHeight - 150) // 6) + 20,
                 40, 40, fill=fill, border='black')
    fill = 'grey' if app.selected == None else rgb(230,0,0)
    if not app.clicked:
        drawRectWithShadow(app, app.width// 2 - 100, app.boxTop + app.boxHeight - 110,
                           200, 50, fill=fill, depth=5)
        change = 0
    else:
        drawRect(app.width // 2 - 105, app.boxTop + app.boxHeight - 115,
                 200, 50, fill=rgb(230, 0, 0))
        change = 5
    drawLabel('Begin', app.width // 2 - change, app.boxTop + app.boxHeight - 85 - change, size=20, fill='white')


def onMousePress(app, mouseX, mouseY):
    if app.selectionActive:
        if (app.width//2 - 100 <= mouseX <= app.width//2 + 100 and
            app.height - 170 <= mouseY <= app.height - 120 and
            app.selected != None):
            app.clicked = True
            app.setupEndSteps = app.steps
            app.selectionActive = False
            app.multiplier = getMultiplier(app.selected)
            app.stopLoss = getMultiplier(app.selected)
            app.takeProfit = getMultiplier(app.selected) * 2
            app.spend = getSpend(app.selected)

        elif (app.boxLeft <= mouseX <= app.boxLeft + app.boxWidth and
              app.boxTop <= mouseY <= app.boxTop + app.boxHeight - 150):
            newSelected = (mouseY - app.boxTop) // ((app.boxHeight - 150) // 3)
            app.selected = None if app.selected == newSelected else newSelected


def getMultiplier(idx):
    if idx == 0: return 0.0001
    if idx == 1: return 0.0002
    if idx == 2: return 0.0003


def getSpend(idx):
    if idx == 0: return 0.05
    if idx == 1: return 0.1
    if idx == 2: return 0.15


def drawSplash(app):
    drawImage(app.backgroundImg, 0, 0)
    drawImage(app.logoImg, app.width // 2, app.height // 2 - 70, align='center')
    drawLabel('ERUCA', app.width // 2, app. height // 2 + 200, font='cinzel',
              size = 72, fill='white')
    drawLoadingBar(app, 200, app.height // 2 + 120, app.steps, 5)


def drawTraining(app):
    drawImage(app.backgroundImg, 0, 0)
    drawRectWithShadow(app, app.width // 2 - app.boxWidth // 2, app.height // 2 - app.boxHeight //2,
                       app.boxWidth, app.boxHeight)

    drawLabel('Welcome to Eruca!', app.width // 2, 120, size=50, font='cinzel')
    drawLabel('Please hold on while historical price information is being analyized',
              app.width // 2, 160, size=16, font='montserrat')
    percentage = pythonRound(app.pointsAnalysed/app.dataLen*100)
    drawLabel(f'{percentage}%', app.width // 2, app.height // 2 - 120, size=90)
    drawLoadingCircle(app, app.width // 2, app.height // 2 + 100, 150, app.pointsAnalysed, app.dataLen)


def drawRectWithShadow(app, left, top, width, height, fill='white', depth=10):
    drawRect(left - depth, top + depth, width, height, fill='black', opacity=40)
    drawRect(left, top, width, height, fill=fill)


def drawLoadingBar(app, barWidth, top, current, total):
    endOpacity = 60 if current != total else 100
    dX = pythonRound(barWidth * (current/total))
    drawLine(app.width // 2 - (barWidth // 2), top,
             (app.width // 2 - (barWidth // 2)) + dX, top,
             lineWidth=10, fill='white', opacity=100)
    drawLine(app.width // 2 - (barWidth // 2) + dX, top,
             app.width // 2 + (barWidth // 2), top,
             lineWidth=10, fill='white', opacity=60)
    drawArc(app.width // 2 - 100, app.height // 2 + 120, 10, 10, 180, 180, fill='white', opacity=100)
    drawArc(app.width // 2 + 100, app.height // 2 + 120, 10, 10, 0, 180, fill='white', opacity=endOpacity)


def drawLoadingCircle(app, centerX, centerY, radius, current, total):
    width, height = radius * 2, radius * 2
    drawCircle(centerX, centerY, radius, fill=rgb(230,0,0), opacity=60)
    sweepAngle = pythonRound(360 * (current / total))
    drawArc(centerX, centerY, width, height, 0, sweepAngle, fill=rgb(230,0,0))


def onStep(app):
    app.steps += 1
    if app.steps == 6:
        app.balance = startBalance()
        app.balanceBTC = 0
        app.openTrades = [] # price, amount, time
        app.closedTrades = [] # purchase price, sell price, amount, time
        app.screen = 'model-train'
        app.data = list(yf.download(tickers='BTC-USD', period='1d', interval='1m')['Close'])[:-1]
        app.dataLen = 50
        app.dataLoaded = True
        trainModel(app)
    if app.screen == 'model-train' and not app.modelLoaded:
        if app.pointsAnalysed == app.dataLen:
            app.trainEndSteps = app.steps
            app.modelLoaded = True
        elif app.pointsAnalysed < app.dataLen: app.pointsAnalysed += 1
    elif app.screen == 'model-train' and app.modelLoaded:
        app.screen = 'risk-select'
        app.selectedBox = None
        app.selectionActive = True
        app.selected = None
        app.setupEndSteps = 0
    elif app.screen == 'risk-select' and app.steps == app.setupEndSteps + 2:
        app.screen = 'main'
        app.intervalView = '1h'
    if app.screen == 'main':
        if app.waitSteps <= 0:
            checkNewData(app)
            updateGrid(app)
            app.waitSteps = app.stepsPerSecond * 30
        else:
            app.waitSteps -= 1


def checkNewData(app):
    newData = list(yf.download(tickers='BTC-USD', period='1d', interval='1m')['Close'])
    if newData[-2] != app.data[-1]:
        app.data.append(newData[-2])
        app.newDataFound = True
        checkTx(app)


def checkTx(app):
    cur = app.displayedVals[app.forward]
    preds = app.displayedVals[:app.forward]
    for pred in preds:
        if pred >= cur * (1 + app.multiplier):
            spend = app.balance * app.spend
            app.balance -= spend
            BTCPurchased = spend/cur
            app.balanceBTC += BTCPurchased
            time = datetime.now().strftime("%H:%M")
            app.openTrades.append([cur, BTCPurchased, time])
            break

    i = 0
    while i < len(app.openTrades):
        if (cur <= app.openTrades[i][0] * (1 - app.stopLoss)
            or cur >= app.openTrades[i][0] * (1 + app.takeProfit)):
            closedTrade = app.openTrades.pop(i)
            closedTrade.insert(1, cur)
            closedTrade[3] = datetime.now().strftime("%H:%M")
            app.balance += closedTrade[2]*cur
            app.balanceBTC -= closedTrade[2]
            app.closedTrades.append(closedTrade)
        i += 1


def updateGrid(app, rerun=True):
    app.displayedVals = app.data[-1:-59:-3]
    app.minDisplayed, app.maxDisplayed = min(app.displayedVals), max(app.displayedVals)
    app.graphCols = len(app.displayedVals)

    graphTimeInterval = 3
    recentHour = datetime.now().hour
    recentMin = datetime.now().minute
    app.colLabels = [f'{recentHour:02}:{recentMin:02}']
    for i in range(app.graphCols - 1):
        recentMin -= graphTimeInterval
        if recentMin < 0:
            recentMin = 60 + recentMin
            recentHour -= 1
            if recentHour < 0: recentHour = 11
        app.colLabels.insert(0, f'{recentHour:02}:{recentMin:02}')

    difference = app.maxDisplayed - app.minDisplayed
    differenceDigitCount = countDigits(difference)
    differenceFirstDigit = getFirstDigit(difference) + 1
    app.graphRows = pythonRound(differenceFirstDigit) + 2
    if app.graphRows < 5: app.graphRows = 5
    app.graphMin = app.minDisplayed // (10**differenceDigitCount) * (10**differenceDigitCount)
    app.graphMax = app.graphMin + ((app.graphRows-1) * 10**differenceDigitCount)
    app.rowHeight = app.graphHeight // (app.graphRows - 1)
    app.rowLabels = [app.graphMin + (i * 10**differenceDigitCount) for i in range(app.graphRows)]

    app.yPlots = [(app.displayedVals[i] - app.graphMin) / ((app.graphRows-1) * 10**differenceDigitCount) for i in range(app.graphCols)]
    app.yPlots = [app.height - 100 - (plot * app.graphHeight) for plot in app.yPlots]
    app.yPlots = app.yPlots[::-1]

    app.colWidth = app.graphWidth / (app.graphCols - 1)
    if rerun: updateGrid(app, rerun=False)

def countDigits(n):
    n = abs(n)
    if n < 10: return 1
    return math.floor(math.log10(n))


def getFirstDigit(n):
    n = abs(n)
    while n > 10:
        n //= 10
    return n


def processData(app):
    inputs, targets = [], []
    dataCopy = copy.deepcopy(app.data)
    chunkSize = app.prev + app.forward
    while len(dataCopy) >= chunkSize:
        currentChunk = dataCopy[-(chunkSize):]
        dataCopy = dataCopy[:-(chunkSize)]
        inputs.append(currentChunk[:app.prev])
        targets.append(currentChunk[app.prev:])

    return inputs, targets

def trainModel(app):
    if not app.modelLoaded: app.inputList, app.targetsList = processData(app) #first train
    if app.modelLoaded and app.newPoints >= app.prev + app.forward:
        app.inputList.extend(app.data[-(app.prev + app.forward):-app.prev])
        app.targetList.extend(app.data[-app.prev:])
    epochTargets = []
    epochs = app.iter

    for _ in range(epochs):
        for i in range(app.future):
            epochTargets += [app.inputList[i]]
            LSTM.updateNodes(epochTargets, app.nodes, app.layers, app.params)
        app.loss = LSTM.updateLoss(app.nodes, app.layers, app.targetsList, app.dataLen)
        app.params = LSTM.updateParams(app.params)
        epochTargets = []
        app.displayedVals[:app.forward] = app.nodes[-1].h[0]


def updateModel(app):
    oldInputList = app.inputList[0]
    newInputList = oldInputList + [app.predsList.pop(0)]
    app.inputList = [newInputList[:] for _ in range(app.futureDataPoints)]
    app.predsList.append(app.data[-1])
    inputs = []

    for i in range(app.futureDataPoints):
        inputs += [app.inputList[i]]
        LSTM.updateNodes(inputs, app.nodes, app.layers, app.params)
    app.params, app.loss = LSTM.updateLoss(app.params, app.nodes, app.layers, app.predsList, app.dataLen)

    app.displayedVals[:app.forward] = app.nodes[-1].h[0]


def main():
    runApp()

main()
