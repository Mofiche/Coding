package AStar;

import com.sun.deploy.util.ArrayUtil;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

class Utils {

    static Spot[] add(Spot[] arr, Spot spot){
        Spot[] ret = new Spot[arr.length +1];
        for(int i = 0; i < ret.length; i++){
            if(i != ret.length -1)ret[i] = arr[i];
            else ret[i] = spot;
        }
        return ret;
    }

    static Spot[] remove(Spot[] input, Spot elem){
        Spot[] res = new Spot[0];
        for(Spot i : input){
            if(!elem.equals(i)){
                res = add(res,i);
            }
        }
        return  res;
    }

    static double heuristic(Spot one , Spot two){
       double a =  abs(one.i - two.i) + abs(one.j - two.j);
       double b = sqrt(((one.i - two.i)*(one.i-two.i))+((one.j-two.j)*(one.j-two.j)));
        return (a+b);
    }

}
