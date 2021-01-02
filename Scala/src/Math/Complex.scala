package Math
import Util._

class Complex(var real : Double = 0, var imaginary : Double = 0) {

  real = round(real)
  imaginary = round(imaginary)

  def re: Double = round(real)
  def im: Double = round(imaginary)


  override def toString: String = "" + re + (if (im < 0) "" else "+") + im + "i"

  def + ( c : Complex) = new Complex(re + c.re,im + c.im)
  def + ( c : Double) = new Complex(re +c , im)
  def - ( c : Complex) = new Complex(re - c.re,im - c.im)
  def - ( c : Double) = new Complex(re -c ,im)
  def * ( c : Complex) = new Complex((re*c.re) - (im*c.im),(re*c.im)+(im*c.re))
  def * ( c : Double) = new Complex(re *c,im * c)
  def / ( c : Complex) = new Complex((re*c.re + im*c.im)/(c.re * c.re + c.im * c.im),((im*c.re)-(re*c.im))/((c.re*c.re) + (c.im*c.im)))
  def / ( c : Double) = new Complex( re / c, im/c)
  private def / (x : Double, c : Complex) : Complex = new Complex(x,0) / c


  def abs: Double = math.sqrt(re*re + im*im)
  def konj = new Complex(re , -im)
  def phi : Double = round(math.atan(im / re))
  def pow( c : Double): Complex = expi(new Complex(0,c*phi)) * math.pow(abs,c)
  def pow( c : Complex): Complex = expi( c * ln)
  def ln = new Complex(math.log(abs),phi)
  def log = ln / math.log(10)
  def log(x : Complex) = ln / x.ln
  def sqrt(x : Double) : Complex = pow(1/x)
  def sqrt(x : Complex) : Complex = pow(/(1,x))




}
