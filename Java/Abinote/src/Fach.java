import java.util.ArrayList;

public class Fach {

    static final ArrayList<Double> Deutsch = new ArrayList<Double>(4);


    static final double[] Geograhie = {9,7,9,9};
    static final double[] Sport = {9,7,9,9};
    static final double[] Mathematik = {13,14,14,13};
    static final double[] Ethik = {10,8,9,9};
    static final double[] Chemie = {13,14,14,13};
    static final double[] Englisch = {9,10,7,7};
    static final double[] Biologie = {10,12,12,14};
    static final double[] Geschichte = {7,5,9,9};
    static final double[] Physik = {13,14,12,13};
    static final double[] Gemeinschaftskunde = {10,10,10,12};
    static final double[] Musik = {10,7,12,5};



    static public int maxNumber(double[] fach){
        double a = fach[0];
        double b = fach[1];
        double c = fach[2];
        double d = fach[3];

        double max = Math.max(Math.max(a,b),Math.max(c,d));
        if(max == a) return 0;
        else if(max == b) return 1;
        else if(max == c) return 2;
        else return 3;
    }



}
