package Notenberechnung


import javax.swing.JOptionPane
import org.apache.commons.lang3.ArrayUtils


class Noten {

  var noten: Array[Double] = Array[Double](0)
  var buffer = 0

  private def isLegit(x : Double): Boolean ={
    if(x <= 15 && x>=0) true
    else false
  }

  private def isArrayLegit(x : Array[Double]) : Boolean = {
    var count = 0
    for(i <- x.indices) {
      if(isLegit(x(i))) count += 1
    }
    if(count == x.length) true
    else false
  }

  def add( x : Double) : Unit = {
    if(isArrayLegit(noten)) {
      if (noten.length == 1 && buffer == 0) {
        noten(0) = x
        buffer += 1
      } else {
        noten = noten :+ x
        buffer = 0
      }
    }else throw CustomException("Array isnt legit")
  }

  def addArray( x : Array[Double]) : Unit = {
    if(isArrayLegit(noten)){

      if(isArrayLegit(x)){

        for(i <- x.indices) add(x(i))

      }else throw CustomException("Inserted Array isnt legit")

    }else throw CustomException("Array isnt legit")
  }

  def del ( x : Double) : Unit = {
    if(isArrayLegit(noten)){
      if(isLegit(x)){
        if(noten contains x){
            noten = ArrayUtils.remove(noten,noten.indexOf(x))
        }else JOptionPane.showMessageDialog(null,"Die Note ist nicht eingetragen gewesen")
      }else JOptionPane.showMessageDialog(null,"Die Note ist nicht im Bereich zwischen 0 und 15")
    }else throw CustomException("Array isnt legit")
  }

  def update ( old : Double, neu : Double) : Unit = {
    del(old)
    add(neu)
  }

  protected[Notenberechnung] def print() : Unit = noten foreach println

  def getArray : Array[Double] = noten

  def average() : Int = {
    if (isArrayLegit(noten)) {
      var sum  : Double = 0
      for (x <- noten.indices) sum += noten(x)
      math.round(sum / noten.length).toInt
    }
    else throw CustomException("Array isnt legit")
  }
}


