package neural

import java.util

class TrainingSet(inputs : util.ArrayList[Integer], goodOutput : util.ArrayList[Double]) {

  def getInputs(): util.ArrayList[Integer] =  inputs
  def getGoodOutputs(): util.ArrayList[Double] =  goodOutput
}
