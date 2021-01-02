package data

import java.util

import neural.TrainingSet

class ReadWriteFile {

  def readTrainingSets() : util.ArrayList[TrainingSet] = {
    var trainingSets  = new util.ArrayList[TrainingSet]()

    for(i <- 0 to 26){
      var letterValue : Char = (i + 65).toChar
      var letter : String   = String.valueOf(letterValue)
      for (list <- readFromFile("/" + letter + ".txt")) {
        trainingSets.add(new TrainingSet(list, GoodOutputs.getInstance.getGoodOutput(letter)))
      }
    }
    trainingSets
  }

  def readFromFile(filename: String) : util.ArrayList[util.ArrayList[Integer]] = {
    var inputs : util.ArrayList[util.ArrayList[Integer]] = new util.ArrayList[util.ArrayList[Integer]]()

  }



}
