# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import tp2, classification
import json
import numpy as np
import matplotlib.pyplot as plt
import nltk

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(591, 577)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.query = QtWidgets.QLineEdit(self.centralwidget)
        self.query.setGeometry(QtCore.QRect(20, 20, 361, 25))
        self.query.setText("")
        self.query.setObjectName("query")
        self.searchButton = QtWidgets.QPushButton(self.centralwidget)
        self.searchButton.setGeometry(QtCore.QRect(400, 20, 89, 25))
        self.searchButton.setObjectName("searchButton")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(23, 89, 371, 411)) 
        self.textEdit.setObjectName("textEdit")
        self.radioPertinences = QtWidgets.QRadioButton(self.centralwidget)
        self.radioPertinences.setGeometry(QtCore.QRect(140, 510, 131, 23))
        self.radioPertinences.setChecked(False)
        self.radioPertinences.setObjectName("radioPertinences")
        self.displayGroup = QtWidgets.QButtonGroup(MainWindow)
        self.displayGroup.setObjectName("displayGroup")
        self.displayGroup.addButton(self.radioPertinences)
        self.radioWeight = QtWidgets.QRadioButton(self.centralwidget)
        self.radioWeight.setGeometry(QtCore.QRect(20, 510, 112, 23))
        self.radioWeight.setChecked(True)
        self.radioWeight.setObjectName("radioWeight")
        self.displayGroup.addButton(self.radioWeight)
        self.radioScalar = QtWidgets.QRadioButton(self.centralwidget)
        self.radioScalar.setGeometry(QtCore.QRect(410, 70, 141, 23))
        self.radioScalar.setChecked(True)
        self.radioScalar.setObjectName("radioScalar")
        self.pertinenceGroup = QtWidgets.QButtonGroup(MainWindow)
        self.pertinenceGroup.setObjectName("pertinenceGroup")
        self.pertinenceGroup.addButton(self.radioScalar)
        self.radioCosine = QtWidgets.QRadioButton(self.centralwidget)
        self.radioCosine.setGeometry(QtCore.QRect(410, 110, 141, 23))
        self.radioCosine.setObjectName("radioCosine")
        self.pertinenceGroup.addButton(self.radioCosine)
        self.radioJacc = QtWidgets.QRadioButton(self.centralwidget)
        self.radioJacc.setGeometry(QtCore.QRect(410, 150, 141, 23))
        self.radioJacc.setObjectName("radioJacc")
        self.pertinenceGroup.addButton(self.radioJacc)
        self.radioBM25 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioBM25.setEnabled(True)
        self.radioBM25.setGeometry(QtCore.QRect(410, 190, 141, 23))
        self.radioBM25.setObjectName("radioBM25")
        self.pertinenceGroup.addButton(self.radioBM25)
        self.radioByTerm = QtWidgets.QRadioButton(self.centralwidget)
        self.radioByTerm.setGeometry(QtCore.QRect(20, 60, 112, 23))
        self.radioByTerm.setChecked(True)
        self.radioByTerm.setObjectName("radioByTerm")
        self.queryTypeGroup = QtWidgets.QButtonGroup(MainWindow)
        self.queryTypeGroup.setObjectName("queryTypeGroup")
        self.queryTypeGroup.addButton(self.radioByTerm)
        self.radioBool = QtWidgets.QRadioButton(self.centralwidget)
        self.radioBool.setEnabled(True)
        self.radioBool.setGeometry(QtCore.QRect(410, 230, 141, 23))
        self.radioBool.setObjectName("radioBool")
        self.pertinenceGroup.addButton(self.radioBool)
        self.displayClusterButton = QtWidgets.QPushButton(self.centralwidget)
        self.displayClusterButton.setGeometry(QtCore.QRect(440, 390, 121, 25))
        self.displayClusterButton.setObjectName("displayClusterButton")
        self.dbscanButton = QtWidgets.QPushButton(self.centralwidget)
        self.dbscanButton.setGeometry(QtCore.QRect(440, 360, 121, 25))
        self.dbscanButton.setObjectName("dbscanButton")
        self.epsilon = QtWidgets.QLineEdit(self.centralwidget)
        self.epsilon.setGeometry(QtCore.QRect(440, 280, 121, 25))
        self.epsilon.setObjectName("epsilon")
        self.minNeighors = QtWidgets.QLineEdit(self.centralwidget)
        self.minNeighors.setGeometry(QtCore.QRect(440, 320, 121, 25))
        self.minNeighors.setObjectName("minNeighors")
        self.radioByDoc = QtWidgets.QRadioButton(self.centralwidget)
        self.radioByDoc.setGeometry(QtCore.QRect(130, 60, 131, 23))
        self.radioByDoc.setObjectName("radioByDoc")
        self.queryTypeGroup.addButton(self.radioByDoc)
        self.radioByQuery = QtWidgets.QRadioButton(self.centralwidget)
        self.radioByQuery.setGeometry(QtCore.QRect(270, 60, 112, 23))
        self.radioByQuery.setObjectName("radioByQuery")
        self.queryTypeGroup.addButton(self.radioByQuery)
        self.naiveBayesButton = QtWidgets.QPushButton(self.centralwidget)
        self.naiveBayesButton.setGeometry(QtCore.QRect(440, 470, 121, 25))
        self.naiveBayesButton.setObjectName("naiveBayesButton")
        self.displayLabelsButton = QtWidgets.QPushButton(self.centralwidget)
        self.displayLabelsButton.setGeometry(QtCore.QRect(440, 500, 121, 25))
        self.displayLabelsButton.setObjectName("displayLabelsButton")
        self.KSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.KSpinBox.setGeometry(QtCore.QRect(480, 190, 51, 26))
        self.KSpinBox.setMinimum(1.2)
        self.KSpinBox.setMaximum(2.0)
        self.KSpinBox.setSingleStep(0.05)
        self.KSpinBox.setObjectName("KSpinBox")
        self.BSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.BSpinBox.setGeometry(QtCore.QRect(530, 190, 51, 26))
        self.BSpinBox.setDecimals(2)
        self.BSpinBox.setMinimum(0.5)
        self.BSpinBox.setMaximum(0.75)
        self.BSpinBox.setSingleStep(0.05)
        self.BSpinBox.setObjectName("BSpinBox")


        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        with open("frequencies.json", "r") as fp:
            self.docs = json.load(fp)

        with open("queries.json", "r") as fp:
            self.queries = json.load(fp)

        with open("relevantDocs.json", "r") as fp:
            self.relevantDocs = json.load(fp)

        self.searchButton.clicked.connect(self.search)
        self.displayClusterButton.clicked.connect(self.displayDBSCAN)
        self.dbscanButton.clicked.connect(self.runDBSCAN)
        self.dataset = None
        self.naiveBayesButton.clicked.connect(self.runNaiveBayes)
        self.displayLabelsButton.clicked.connect(self.displayBayes)
        self.queryFeatures = None
        self.queryLabels = None

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Recherche D\'information"))
        self.searchButton.setText(_translate("MainWindow", "Search"))
        self.radioPertinences.setText(_translate("MainWindow", "Pertinences"))
        self.radioWeight.setText(_translate("MainWindow", "Freq et Poids"))
        self.radioScalar.setText(_translate("MainWindow", "Scalar product"))
        self.radioCosine.setText(_translate("MainWindow", "Cosine measure"))
        self.radioJacc.setText(_translate("MainWindow", "Jaccard measure"))
        self.radioBM25.setText(_translate("MainWindow", "BM25"))
        self.radioByTerm.setText(_translate("MainWindow", "Par termes"))
        self.radioBool.setText(_translate("MainWindow", "Boolean"))
        self.displayClusterButton.setText(_translate("MainWindow", "Display Clusters"))
        self.dbscanButton.setText(_translate("MainWindow", "Run DBSCAN"))
        self.epsilon.setText(_translate("MainWindow", "Epsilon"))
        self.minNeighors.setText(_translate("MainWindow", "MinNeighbors"))
        self.radioByDoc.setText(_translate("MainWindow", "Par documents"))
        self.radioByQuery.setText(_translate("MainWindow", "Par Query"))
        self.naiveBayesButton.setText(_translate("MainWindow", "Naive Bayes"))
        self.displayLabelsButton.setText(_translate("MainWindow", "Display labels"))


    
    def search(self):
        # query = self.query.text().split()
        ExpReg = nltk.RegexpTokenizer(r'(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*') # \d : équivalent à [0-9]
        query = ExpReg.tokenize(self.query.text())

        # Search by termes
        if self.radioByTerm.isChecked() :
            weights, frenquencies = tp2.weight(self.docs, query)

            if self.radioWeight.isChecked():
                self.textEdit.setText("""Terme                   Doc                   Freq                   Poids\n""")
                for terme in weights.keys():
                    for doc in weights[terme].keys():
                        if frenquencies[terme][doc] > 0:
                            self.textEdit.append(terme + "\t" + doc + "\t" + str(frenquencies[terme][doc]) + "\t" + str("%.4f" % weights[terme][doc])+ "\n")

            else:
                self.textEdit.setText("")

                if self.radioScalar.isChecked():
                    pertinence = tp2.scalarProduct(weights)
                
                elif self.radioBool.isChecked():
                    pertinence = tp2.execRequest(self.query.text(), self.docs)
                    if pertinence is None:
                        self.textEdit.setText("ERROR")
                        return

                    for doc in pertinence.keys():
                        if pertinence[doc] > 0:
                            self.textEdit.append(doc + "\t 1")
                    return

                else:
                    measure = {"Jaccard measure" : tp2.jaccardMeasure, "Cosine measure" :tp2.cosineMeasure, "BM25" : tp2.BM25}[self.pertinenceGroup.checkedButton().text()]
                    if measure != tp2.BM25:
                        pertinence = measure(self.docs, query)
                    else:
                        k = self.KSpinBox.value()
                        b = self.BSpinBox.value()
                        pertinence = measure(self.docs, query, b, k)
                
                pertinence = dict(sorted(pertinence.items(), key=lambda item: item[1], reverse=True))
                for doc in pertinence.keys():
                    self.textEdit.append(doc + "\t" + str("%.6f" % pertinence[doc]))

        #Search by Docs     
        elif self.radioByDoc.isChecked():
            frenquencies = {}
            self.textEdit.setText("""Doc                   Terme                   Freq                   Poids\n""")
            termes = set()
            for doc in query:
                doc = doc.capitalize()
                frenquencies[doc] = self.docs[doc]
                termes |= frenquencies[doc].keys()

            weights = tp2.weight(self.docs, list(termes), stem=False)[0]
            for doc in query:
                doc = doc.capitalize()
                for terme in frenquencies[doc].keys():
                    self.textEdit.append(doc + "\t" + terme + "\t" + str(frenquencies[doc][terme]) + "\t" + str("%.4f" % weights[terme][doc])+ "\n")

        #Search by Query
        else:
            self.textEdit.setText("Query                   Doc                   Relevance")

            for q in query:
                q = q.capitalize()
                weights = tp2.weight(self.docs, self.queries[q], stem=False)[0]

                if self.radioScalar.isChecked():
                    pertinence = tp2.scalarProduct(weights)
                
                else:
                    measure = {"Jaccard measure" : tp2.jaccardMeasure, "Cosine measure" :tp2.cosineMeasure, "BM25" : tp2.BM25}[self.pertinenceGroup.checkedButton().text()]
                    if measure != tp2.BM25:
                        pertinence = measure(self.docs, self.queries[q])
                    else:
                        k = self.KSpinBox.value()
                        b = self.BSpinBox.value()
                        pertinence = measure(self.docs, self.queries[q], b, k) 
                
                pertinence = dict(sorted(pertinence.items(), key=lambda item: item[1], reverse=True))
                for doc in pertinence.keys():
                    self.textEdit.append(q + "\t" + doc + "\t" + str("%.6f" % pertinence[doc]))
                
                self.textEdit.append("\n")

                correctDocs = 0
                x = []
                y = []
                i = 1
                for relevantDoc in pertinence.keys() :
                    if relevantDoc[1:] in self.relevantDocs[q]:
                        correctDocs = correctDocs + 1
                    
                    if i == 5:
                        correctDocs5 = correctDocs
                    if i == 10:
                        correctDocs10 = correctDocs
                    if i == len(self.relevantDocs[q]):
                        correctDocsR = correctDocs

                    y.append(correctDocs / i)
                    x.append(correctDocs)
                    i = i + 1

                plt.plot( np.array(x) / correctDocs ,y, label=q)
                
                precision = correctDocs / len(pertinence.keys())
                p5 = correctDocs5 / 5 
                p10 = correctDocs10 / 10 
                rappel = correctDocs / len(self.relevantDocs[q])
                fmeasure = (2*rappel*precision) / (precision + rappel)
                r_precision = correctDocsR / len(self.relevantDocs[q])

                self.textEdit.append("Precision = " + str(correctDocs) + "/" + str(len(pertinence.keys())) + " = " + str(precision))
                self.textEdit.append("P5 = " + str(correctDocs5) + "/" + str(5) + " = " + str(p5))
                self.textEdit.append("P10 = " + str(correctDocs10) + "/" + str(10) + " = " + str(p10))
                self.textEdit.append("R-Precision = " + str(correctDocsR) + "/" + str(len(self.relevantDocs[q])) + " = " + str(r_precision))
                self.textEdit.append("Rappel = " + str(correctDocs) + "/" + str(len(self.relevantDocs[q])) + " = " + str(rappel))
                self.textEdit.append("F-Measure = " + str(fmeasure))

                self.textEdit.append("\n\n\n\n")
            
            plt.title("Courbe(s) Rappel-Precision")
            plt.legend()
            plt.show()


    def displayDBSCAN(self):
        clusters = np.genfromtxt("clusters.csv",delimiter=",", dtype=np.int8)
        self.textEdit.setText("")

        for i in range(len(self.docs)):
            self.textEdit.append("I" + str(i+1) + "\t" + str(clusters[i]))

    def runDBSCAN(self):
        if self.dataset is None:
            self.dataset = np.genfromtxt("weights.csv", delimiter=",", dtype = np.float32)
        minNeighbors = int(self.minNeighors.text())
        eps = float(self.epsilon.text())
        classification.dbscan(self.dataset, eps, minNeighbors)
        self.displayDBSCAN()
    

    def displayBayes(self):
        self.textEdit.setText("")
        for i in range(len(self.queryLabels)):
            self.textEdit.append("Q" + str(i+1) + "\t" + str(self.queryLabels[i]))

    def runNaiveBayes(self):
        if self.queryFeatures is None :
            self.queryFeatures = np.genfromtxt("queryFeatures.csv", delimiter=",", dtype = np.int8)

        self.queryLabels = classification.bayesianInference(self.queryFeatures)
        self.displayBayes()
        



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
