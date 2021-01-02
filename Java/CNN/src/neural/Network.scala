package neural

import java.util
import java.util.ArrayList

class Network {

  private var neurons : util.ArrayList[Neuron] = new util.ArrayList[Neuron]()

  def addNeurons(count : Int) : Unit = for (i <- count) neurons.add(new Neuron())

  def setInputs(inputs : util.ArrayList[Integer]) : Unit = {
    for (n <- neurons.size()) neurons.get(n).setInputs(inputs)
  }

  def getOutputs(): util.ArrayList[Double] = {
    var outputs = new util.ArrayList[Double]()
    for( i <- neurons.size()) outputs.add(neurons.get(i).getOutput())
    outputs
  }

  def adjustWages( goodOutput : util.ArrayList[Double]) : Unit = {
    for( i <- neurons.size()){
      var delta : Double = goodOutput.get(i) - neurons.get(i).getOutput()
      neurons.get(i).adjustWeights(delta)
    }
  }
}
