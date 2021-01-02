import requests

class KURSNAME:
    dax = "index/dax"
    ftse_100 = "index/ftse_100"
    concentra = "fonds/concentra"
    industria = "fonds/industria"
    kapitalplus = "fonds/kapital_plus"


class KURS:
    name = ""

    def __init__(self, name):
        self.name = name

    def getKurs(self):
        try:
            page = requests.get('https://www.finanzen.net/' + self.name)
            content = page.text
            b = content.find('<div class="col-xs-5 col-sm-4 text-sm-right text-nowrap">')
            kurs = []
            for i in range(b + 59, b + 70):
                kurs.append(content[i])
            if "." in kurs:
                kurs.remove('.')
            kurs = ''.join(kurs).replace(',', '.')
            kurs = kurs.split(".")[0]
            kommastellen = []
            try:
                for i in range(b + 61 + len(kurs), b + 63 + len(kurs)):
                    kommastellen.append(content[i])
                kommastellen = ''.join(kommastellen)
                kommastellen = float(kommastellen)

            except ValueError:
                kommastellen = []
                for i in range(b + 60 + len(kurs), b + 62 + len(kurs)):
                    kommastellen.append(content[i])
                kommastellen = ''.join(kommastellen)
                kommastellen = float(kommastellen)

            kurs = float(kurs) + kommastellen / 100
            return kurs
        finally:
            pass
