package Math

import Util._

object main{

  def main(args: Array[String]): Unit = {
      val x = math.pow(10, 6).toInt
      var a = for (i <- 0 to x) yield i

      var beginn = System.currentTimeMillis()

      var b = for (i <- a) yield isprime(a(i))

      var end = System.currentTimeMillis()
      println("Alle Primzahlen von bis 0 bis " + x + " :")
      for (j <- a) if (b(j)) {
        println(a(j))
      }
      println("Zeit : " + (end - beginn) + " ms")


    }

}