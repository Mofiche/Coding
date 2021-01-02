package Morse;

import java.io.*;

public class Test  extends Utils implements Serializable{

    public static void main(String[] args) {

        String input = "Julia";
        if(args.length == 1){
            input = args[0];
        }else if(args.length > 2){
            System.out.println("Please enter only one word || Bitte nur ein Wort eingeben");
        }else{
            input = "Test";
        }
        String filename = "src/Morse/morse.ser";
        File file = new File(filename);
        MorseStringConverter conv = new MorseStringConverter();

        int[][] morse = conv.String2Morse(input);
        print(morse);

        // save(file, morse);
        // int[][] loadedMorse = load(file);

        String output = conv.Morse2String(morse);
        System.out.println(output);

    }

}
