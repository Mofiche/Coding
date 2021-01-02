package Complex;

public class Test {

    private static double round(double a){
        return Math.round(1000.0 * a) / 1000.0;
    }


    public static void main(String[] args){

        final double PI = 3.14159265358979323846;

        Complex a = new Complex(3,5);
        Complex b = new Complex(-5,9);
        Complex c = new Complex(0,1);
        Complex d = Complex.multiply(a,b);

        d.print();

    }

}
