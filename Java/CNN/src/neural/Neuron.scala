package neural
import java.util


class Neuron {

  def sigmoidValue(arg : Double): Double = 1 / (1 + math.exp(-arg))

  private var BIAS : Int = 1
  private var LEARNING_RATIO : Double = 0.1

  private var inputs : util.ArrayList[Integer] = new util.ArrayList[Integer]()
  private var weights : util.ArrayList[Double] = new util.ArrayList[Double]()
  private var biasWeight : Double = 0
  private var output : Double = 0

  this.biasWeight = Math.random()

  def setInputs(inputs : util.ArrayList[Integer]): Unit = {
    if( this.inputs.size() == 0){
      this.inputs = new util.ArrayList[Integer](inputs)
      generateWeights()
    }
    this.inputs = new util.ArrayList[Integer](inputs)
  }

  private def generateWeights() = for(_ <- inputs.size()) weights.add(Math.random())

  def calculateOutput(): Unit = {
    var sum: Double = 0
    for (i <- inputs.size()) sum += inputs.get(i) * weights.get(i)
    sum += BIAS * biasWeight
    output = sigmoidValue(sum)
  }

  def adjustWeights(delta : Double): Unit = {
    for ( i <- inputs.size()){
      var d = weights.get(i)
      d += LEARNING_RATIO * delta * inputs.get(i)
      weights.set(i , d)
    }
    biasWeight += LEARNING_RATIO * delta * BIAS
  }

  def getOutput(): Double = {
    calculateOutput()
    output
  }

}
