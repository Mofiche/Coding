from datetime import date
import time


class Auto:

    def __init__(self, Marke, Farbe, Leistung, Gewicht = 0 ):
        self.Marke=Marke
        self.Farbe=Farbe
        self.Leistung=Leistung
        self.Gewicht=Gewicht

    def setMarke(self, Marke):
        self.Marke = Marke

    def getMarke(self):
        return self.Marke

    def setFarbe(self, Farbe):
        self.Farbe = Farbe

    def getFarbe(self):
        return self.Farbe

    def setLeistung(self,Leistung):
        self.Leistung=Leistung

    def getLeistung(self):
        return self.Leistung

    def setGewicht(self, Gewicht):
        self.Gewicht = Gewicht

    def getGewicht(self):
        return self.Gewicht


SchatzisAuto = Auto("Renault", "Rot", 54, 1111)
neuesAuto = Auto("Mercedes", "Schwarz", 500)
SchatzisAuto.setLeistung(54)
"""
SchatzisAuto.setMarke("Renault")
SchatzisAuto.setFarbe("rot")
SchatzisAuto.setGewicht(1111)
SchatzisAuto.setLeistung(54)
"""
print(SchatzisAuto.getFarbe(), SchatzisAuto.getLeistung())
neuesAuto.setGewicht(888)
print(neuesAuto.getFarbe(), neuesAuto.getGewicht())

