package Math

class Vector(var x : Double, var y: Double, var z : Double = 0) {

  override def toString: String = "(" + x + "|" + y + "|" + z + ")"
  def print(a : Vector = this) = println(a.toString)
  def length: Double = scala.math.sqrt(this dotProdcuct this)
  def abs : Double = length
  def +(a : Vector) : Vector = new Vector(x + a.x , y + a.y, z + a.z)
  def -(a : Vector) : Vector = new Vector(x - a.x , y - a.y, z - a.z)
  def *(a : Double) : Vector = new Vector(x * a , y * a, z * a)
  def /(a : Double) : Vector = new Vector(x / a , y / a, z / a)
  def crossProduct(a : Vector) = new Vector(y*a.z - z*a.y,z*a.x - x*a.z,x*a.y - y*a.x)
  def dotProdcuct(a : Vector) : Double = x*a.x + y*a.y +z *a.z
  def norm: Vector = this / abs
  def angleRAD(a : Vector) : Double = scala.math.acos((this dotProdcuct a)/(abs * a.abs))
  def angleDEG(a : Vector) : Double = angleRAD(a) * (180/scala.math.Pi)
}
