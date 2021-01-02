package Math.Functionplotter

import scalafx.Includes._
import scalafx.application.JFXApp
import scalafx.scene.Scene
import scalafx.scene.paint.Color._
import scalafx.scene.shape.Rectangle

object plotter extends JFXApp {

  def func(i : Double): Double = {
   if( i < 0)( i*i )/1000
   else math.sqrt(i)
  }
  stage = new JFXApp.PrimaryStage {

    var cont = List(
      new Rectangle{
        x = 250
        y = 0
        width = 1
        height = 500
      },
      new Rectangle{
        x = 0
        y = 250
        width = 500
        height = 1
      }
    )

    for(i <- Range(-2500,2500,1)) {
      cont = cont ::: List(new Rectangle {
        x = 250 + i
        y = 250 + (-1*func(i) * 10)
        width = 1
        height = 1
      })
    }

    title.value = "Hello Stage"
    width = 500
    height = 500
    scene = new Scene {
      fill = White
      content = cont
    }
  }
}