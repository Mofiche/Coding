import javax.swing.*;

public class Note {

    String kennung ="";
    int wert=-1;

    Note(String ken, int we){
        if(ken.length() == 3 && we <= 15 && we >= 0){
            kennung = ken;
            wert = we;
        }else{
            JOptionPane.showMessageDialog(null,"Kennung stimmt nicht oder Note nicht zwischen 1 und 15");
        }
    }

}
