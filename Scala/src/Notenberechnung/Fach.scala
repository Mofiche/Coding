package Notenberechnung

class Fach {

  var isLK : Boolean = false
  var Klausuren, LK  = new Noten
  private var KlausurNote : Double = Klausuren.average()
  private var LKNote : Double = LK.average()
  var Endnote : Double = (KlausurNote + LKNote) / 2


}
