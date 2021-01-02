package Notenberechnung

object Utilities{
  def round(x : Double): Double = math.rint(x* math.pow(10,5)) / math.pow(10,5)
  def points2note(x : Double) : Double = round((17 - x) / 3)
  def note2points(x : Double) : Double = round(17 - 3*x)
}
