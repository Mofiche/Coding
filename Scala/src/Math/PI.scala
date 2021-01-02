package Math

import java.math.MathContext

import scala.math.ScalaNumber

object PI {

  def sqrt(x : BigDecimal) : BigDecimal = {
    val epsilon = BigDecimal(1E-15)
    var t = x
    var cont1 = BigDecimal(1)
    var cont2 = BigDecimal(1)
    do {
      t = (x / t + t) / BigDecimal(2)
      cont1 = t- x/t
      cont2 = cont1.abs
    }while(cont2 > epsilon)
    t
  }
  def sqrtd(x : BigDecimal) : BigDecimal = {
    val epsilon = BigDecimal(1E-3)
    var t = x
    var cont1 = BigDecimal(1)
    var cont2 = BigDecimal(1)
    do {
      t = ((x / (t*t)) + BigDecimal(2)*t) / BigDecimal(3)
      cont1 = t - x/t
      cont2 = cont1.abs
    }while(cont2 > epsilon)
    t
  }

  def main(args: Array[String]): Unit = {
    /**for(j <- 1 to 10){
      var a = BigDecimal(1)
      var b = a / sqrt(BigDecimal(2))
      var s = BigDecimal(0.25)
      var A, B, c, S = BigDecimal(1)

      for(i <- 0 to j){
        A = (a + b)/BigDecimal(2)
        B = sqrt(a * b)
        c = (a-A)*(a-A)
        S = s - BigDecimal(2).pow(i)*c

        a = A
        b = B
        s = S
      }

      var PI = (A*A)/s
      println(j + ". :" + (PI.toDouble - scala.math.Pi))
    }

      **/

    var a = sqrtd(BigDecimal(27))
    print(a)

  }
}
