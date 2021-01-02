
public class Test {

   static double f(double x){
       double y = x;
       return y;
    }

    static double streifen(double x, double breite){
        return breite*f(x);
    }

    static double Integral(double a, double b){

        final double pres = 1E8;
        double sum = -1;
        if(b>a){
            double breite = (b-a)/pres;
            double wert;
            sum = 0 ;
            for(int i =0 ; i < pres ; i++){
                wert = a+ breite*i;
                sum += streifen(wert,breite);
            }
        }
        return sum;
    }
    public static void main(String[] args){

    double value = Integral(-5,5);
    System.out.println(value*value);



    }

}
