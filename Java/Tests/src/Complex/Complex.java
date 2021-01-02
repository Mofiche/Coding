package Complex;

public class Complex{

    private final double PI = 3.14159265358979323846;

    double re;
    double im;

    Complex(double real , double imaginary ){
        this.re = real;
        this.im = imaginary;
    }

    private static double round(double a){
        return Math.round(1000.0 * a) / 1000.0;
    }

    void print(){
        if(im >= 0 ) {
            System.out.println(round(re) + "+" + round(im) + "i");
        }else{
            System.out.println(round(re) + "-" + round(-im) + "i");
        }
    }

    static void print(Complex a){
        if(a.im >= 0 ) {
            System.out.println(a.re + "+" + a.im + "i");
        }else{
            System.out.println(a.re + "-" + -a.im + "i");
        }
    }

    double real(){
        return re;
    }

    static double real(Complex a){
        return a.re;
    }

    double imaginary(){
        return im;
    }

    static double imaginary(Complex a){
        return a.im;
    }

    double length(){
        return Math.sqrt((re*re)+(im*im));
    }

    static double length(Complex a){
        return Math.sqrt((a.re*a.re)+(a.im*a.im));
    }

    Complex add(Complex a){
        return new Complex(a.re + re, a.im + im);
    }

    static Complex add(Complex a, Complex b){
        return new Complex(a.re + b.re, a.im + b.im);
    }

    Complex add(double a){
        return new Complex( a +re , im);
    }

    static Complex add(Complex b, double a){
        return new Complex( a +b.re , b.im);
    }

    Complex subtract(Complex a){
        return new Complex(a.re - re, a.im - im);
    }

    static Complex subtract(Complex a, Complex b){
        return new Complex(a.re - b.re, a.im - b.im);
    }

    Complex subtract(double a){
        return new Complex(re - a,im);
    }

    static Complex subtract(Complex a, double b){
        return new Complex(a.re - b, a.im);
    }

    double phi(){
        if(im >= 0 ){
            return Math.acos(re/length());
        }else{
            return -Math.acos(re/length());
        }
    }

    static double phi(Complex a){
        if(a.im >= 0 ){
            return Math.acos(a.re/a.length());
        }else{
            return -Math.acos(a.re/a.length());
        }
    }

    Complex multiply(double a){
        return new Complex(a*re,a*im);
    }

    static Complex multiply(double a, Complex b){
        return new Complex(a*b.re,a*b.im);
    }

    static Complex expi(double a ){
        return new Complex(Math.cos(a), Math.sin(a));
    }

    static Complex expi(Complex a ){
        return Complex.multiply(Math.exp(a.re),expi(a.im));
    }

    Complex multiply(Complex a){
        return Complex.multiply(a.length() * length(),Complex.expi(a.phi() + phi()));
    }

    static Complex multiply(Complex a, Complex b){
        return Complex.multiply(a.length() * b.length(),Complex.expi(a.phi() + b.phi()));
    }

    Complex divideBy(Complex a){
        return Complex.multiply(length()/a.length(),Complex.expi(phi()-a.phi()));
    }

    static Complex divide(Complex a, Complex b){
        return Complex.multiply(a.length()/b.length(),Complex.expi(a.phi()-b.phi()));
    }

    Complex divideBy(double a){
        return new Complex(re / a, im/a);
    }

    static Complex divide(Complex b, double a){
        return new Complex(b.re / a, b.im/a);
    }

    static Complex divide(double n , Complex a){
        return Complex.multiply(n,a.pow(-1));
    }

    Complex conj(){
        return new Complex(re, -im);
    }

    static Complex conj(Complex a){
        return new Complex(a.re, -a.im);
    }

    Complex ln(){
        return new Complex(Math.log(length()),phi());
    }

    static Complex ln( Complex a){
        return new Complex(Math.log(a.length()),a.phi());
    }

    Complex log10(){
        return Complex.divide(ln(),Math.log(10));
    }

    static Complex log10(Complex a){
        return Complex.divide(a.ln(),Math.log(10));
    }

    Complex logn(double n){
        return Complex.divide(ln(),Math.log(Math.abs(n)));
    }

    static Complex logn(Complex b, double n){
        return Complex.divide(b.ln(),Math.log(Math.abs(n)));
    }

    Complex pow(double n){
        return Complex.multiply(Math.pow(length(),n),Complex.expi(n*phi()));
    }

    static Complex pow(Complex a, double n){
        return Complex.multiply(Math.pow(a.length(),n),Complex.expi(n*a.phi()));
    }

    Complex pow(Complex a){
        return Complex.expi(Complex.multiply(a,this.ln()));
    }

    static Complex pow(Complex a, Complex b){
        return Complex.expi(Complex.multiply(a,b.ln()));
    }

    static Complex pow(double n, Complex a){
        return Complex.multiply(Math.exp(a.re * Math.log(n)),Complex.expi(new Complex(0,a.im * Math.log(n))));
    }


    Complex sqrtn(double n){
        return Complex.multiply(Math.pow(length(),1/n),expi(new Complex(0,phi()/n)));
    }

    static Complex sqrt(Complex a, double n){
        return Complex.multiply(Math.pow(a.length(),1/n),expi(new Complex(0,a.phi()/n)));
    }

    Complex sqrtn(Complex n){
        return Complex.pow(this,Complex.divide(1,n));
    }

    static Complex sqrtn(Complex n, Complex m){
        return Complex.expi(Complex.multiply(Complex.divide(1,n),new Complex(Math.log(m.length()),m.phi())));
    }

    Complex sqrt(){
        return sqrtn(2);
    }

    static Complex sqrt(Complex a){
        return a.sqrtn(2);
    }




}
