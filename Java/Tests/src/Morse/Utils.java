package Morse;

import java.io.*;

class Utils {
    static void print(int[][] data){
        for (int[] aData : data) {
            for (int j = 0; j < 5; j++) {
                if (aData[j] == 0 || aData[j] == 1){
                    String length = "";
                    if(aData[j] == 0) length = "short || kurz";
                    else length = "long || lang";
                    System.out.println(length);
                }
            }
            System.out.print("\n");
        }
    }

    static String charArr2String(char[] data){
        String ret = "";
        for (char aData : data) {
            ret = ret.concat("" + aData);
        }
        return  ret ;
    }

    static void save(File file, int[][] object){
        try {
            ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file));
            os.writeObject(object);
            os.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    static int[][] load(File file){
        ObjectInputStream is;
        try{
            is = new ObjectInputStream(new FileInputStream(file));
            return (int[][]) is.readObject();
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }
}
