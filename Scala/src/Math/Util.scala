package Math

import math._

object Util {

  def sqrt(x : Double) : Any = {
    if(x == -1) new Complex(0,1)
    else math.sqrt(x)
  }

  def expi( x : Double) = new Complex(math.cos(x),math.sin(x))
  def expi( c : Complex) : Complex = expi(c.im) * math.exp(c.re)
  def round(x : Double): Double = math.rint(x* math.pow(10,10)) / math.pow(10,10)

  def isprime(x : Int) : Boolean = {
    var a = false
    var b = 3
    if(x <= 2){
      if(x == 2) true else false
    }else{
      if(x % 2 == 0 ){
        a = true
      }else{
        a = false
        while(b*b <= x && !a){
          if( x % b == 0) {
            a = true
          }else{
            b += 2
          }
        }
      }
      !a
    }
  }

  def zeta(x : Double) : Double = {
    if(x == 0) return 0.5
    var y : Double = 0
    for(i <- 1 to math.pow(10,7).toInt) y = y + math.pow(math.pow(i,x),-1)
    y
  }

  def pow( x : Double , y: Complex) : Complex = new Complex(x,0).pow(y)

  def zeta(x : Complex) : Complex = {
    if(x == new Complex(0,0)) return new Complex(0.5,0)
    var y : Complex = new Complex(0,0)
    for(i <- 1 to math.pow(10,5).toInt) y = y + pow(i,x).pow(-1)
    y
  }

  def f(x : Double) : Double = x*x

  def integral(down : Double , up : Double) : Double = {

    var sum : Double = 0
    val pres : Int = 1
    val step : Double = 1 / pres
    var x : Double = down
    for(i <- Range(down.toInt,up.toInt,1)) {
      sum = sum + (step * f(x))
      x = x + step
    }
sum
  }



}
