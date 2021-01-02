package PI;

public class Pi {

    public static void main(String args[]){

       double y = Math.sqrt(2)-1;
       double a = 6 - 4*Math.sqrt(2);
       double Y,A;
       double pi;
       for(int i = 0 ; i < 100 ; i++){
           Y = (1-Math.pow(1-Math.pow(y,4),0.25))/(1+Math.pow(1-Math.pow(y,4),0.25));
           A = (Math.pow(1+Y,4)*a)-(Math.pow(2,2*i -1)*y*(1+y+Math.pow(y,2)));
           pi = 1/a;
           System.out.println(pi);
       }

    }

}
