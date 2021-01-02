
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.stage.Stage;


import javax.swing.*;

public class Main extends Application {

    @Override
    public void start(Stage Stage) throws Exception{
        Parent root = FXMLLoader.load(getClass().getResource("gui.fxml"));
        Stage.setTitle("Notenberechnung");
        Stage.setScene(new Scene(root, 800, 600));
        Stage.setResizable(false);

      //  Button btn = (Button) root.lookup("#btn1");


        Notenberechnung.Noten mathe = new Notenberechnung.Noten();
        mathe.add(15);
        mathe.add(15);
        mathe.add(15);
        mathe.print();

        Stage.show();
    }


    public static void main(String[] args) {
        launch(args);
    }
}
