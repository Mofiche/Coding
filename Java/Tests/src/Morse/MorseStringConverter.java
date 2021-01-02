package Morse;

import java.util.Arrays;

class MorseStringConverter extends Utils {

    // 0 = Kurz
    // 1 = Lang

    private final int[] a = {0,1,-1,-1,-1};
    private final int[] b = {1,0,0,0,-1};
    private final int[] c = {1,0,1,0,-1};
    private final int[] d = {1,0,0,-1,-1};
    private final int[] e = {0,-1,-1,-1,-1};
    private final int[] f = {0,0,1,0,-1};
    private final int[] g = {1,1,0,-1,-1};
    private final int[] h = {0,0,0,0,-1};
    private final int[] i = {0,0,-1,-1,-1};
    private final int[] j = {0,1,1,1,-1};
    private final int[] k = {1,0,1,-1,-1};
    private final int[] l = {0,1,0,0,-1};
    private final int[] m = {1,1,-1,-1,-1};
    private final int[] n = {1,0,-1,-1,-1};
    private final int[] o = {1,1,1,-1,-1};
    private final int[] p = {0,1,1,-1,-1};
    private final int[] q = {1,1,0,1,-1};
    private final int[] r = {0,1,0,-1,-1};
    private final int[] s = {0,0,0,-1,-1};
    private final int[] t = {1,-1,-1,-1,-1};
    private final int[] u = {0,0,1,-1,-1};
    private final int[] v = {0,0,0,1,-1};
    private final int[] w = {0,1,1,-1,-1};
    private final int[] x = {1,0,0,1,-1};
    private final int[] y = {1,0,1,1,-1};
    private final int[] z = {1,1,0,0,-1};
    private final int[] zero = {1,1,1,1,1};
    private final int[] one = {0,1,1,1,1};
    private final int[] two = {0,0,1,1,1};
    private final int[] three = {0,0,0,1,1};
    private final int[] four = {0,0,0,0,1};
    private final int[] five = {0,0,0,0,0};
    private final int[] six = {1,0,0,0,0};
    private final int[] seven = {1,1,0,0,0};
    private final int[] eight = {1,1,1,0,0};
    private final int[] nine = {1,1,1,1,0};
    private final int[] space = {1,0,0,0,1};

    int[][] String2Morse(String s){
        char[] inf = s.toLowerCase().toCharArray();
        int[][] morse = new int[s.length()][5];
        for(int i = 0 ; i < inf.length;i++){
            morse[i] = Char2Morse(inf[i]);
        }
        return morse;
    }

    String Morse2String(int [][] inf){
        char[] out = new char[inf.length];
        for(int i = 0; i< inf.length; i++) out[i] = Morse2Char(inf[i]);
        return charArr2String(out);
    }

    char Morse2Char(int[] data){
        char ret = '|';

        if(Arrays.equals(data, a)) ret = 'a';
        else if(Arrays.equals(data, b)) ret = 'b';
        else if(Arrays.equals(data, c)) ret = 'c';
        else if(Arrays.equals(data, d)) ret = 'd';
        else if(Arrays.equals(data, e)) ret = 'e';
        else if(Arrays.equals(data, f)) ret = 'f';
        else if(Arrays.equals(data, g)) ret = 'g';
        else if(Arrays.equals(data,h)) ret = 'h';
        else if(Arrays.equals(data,i)) ret = 'i';
        else if(Arrays.equals(data,j)) ret = 'j';
        else if(Arrays.equals(data,k)) ret = 'k';
        else if(Arrays.equals(data,l)) ret = 'l';
        else if(Arrays.equals(data,m)) ret = 'm';
        else if(Arrays.equals(data,n)) ret = 'n';
        else if(Arrays.equals(data,o)) ret = 'o';
        else if(Arrays.equals(data,p)) ret = 'p';
        else if(Arrays.equals(data,q)) ret = 'q';
        else if(Arrays.equals(data,r)) ret = 'r';
        else if(Arrays.equals(data,s)) ret = 's';
        else if(Arrays.equals(data,t)) ret = 't';
        else if(Arrays.equals(data,u)) ret = 'u';
        else if(Arrays.equals(data,v)) ret = 'v';
        else if(Arrays.equals(data,w)) ret = 'w';
        else if(Arrays.equals(data,x)) ret = 'x';
        else if(Arrays.equals(data,y)) ret = 'y';
        else if(Arrays.equals(data,z)) ret = 'z';
        else if(Arrays.equals(data,one)) ret = '1';
        else if(Arrays.equals(data,two)) ret = '2';
        else if(Arrays.equals(data,three)) ret = '3';
        else if(Arrays.equals(data,four)) ret = '4';
        else if(Arrays.equals(data,five)) ret = '5';
        else if(Arrays.equals(data,six)) ret = '6';
        else if(Arrays.equals(data,seven)) ret = '7';
        else if(Arrays.equals(data,eight)) ret = '8';
        else if(Arrays.equals(data,nine)) ret = '9';
        else if(Arrays.equals(data,zero)) ret = '0';
        else if(Arrays.equals(data,space)) ret = ' ';

        return ret;
    }

    int[] Char2Morse(char letter){
        int[] ret = new int[5];
        switch (letter){
            case 'a':
                ret[0] = 0;
                ret[1] = 1;
                ret[2] = -1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'b':
                ret[0] = 1;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = 0;
                ret[4] = -1;
                break;

            case 'c':
                ret[0] = 1;
                ret[1] = 0;
                ret[2] = 1;
                ret[3] = 0;
                ret[4] = -1;
                break;

            case 'd':
                ret[0] = 1;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'e':
                ret[0] = 0;
                ret[1] = -1;
                ret[2] = -1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'f':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = 1;
                ret[3] = 0;
                ret[4] = -1;
                break;

            case 'g':
                ret[0] = 1;
                ret[1] = 1;
                ret[2] = 0;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'h':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = 0;
                ret[4] = -1;
                break;

            case 'i':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = -1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'j':
                ret[0] = 0;
                ret[1] = 1;
                ret[2] = 1;
                ret[3] = 1;
                ret[4] = -1;
                break;

            case 'k':
                ret[0] = 1;
                ret[1] = 0;
                ret[2] = 1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'l':
                ret[0] = 0;
                ret[1] = 1;
                ret[2] = 0;
                ret[3] = 0;
                ret[4] = -1;
                break;

            case 'm':
                ret[0] = 1;
                ret[1] = 1;
                ret[2] = -1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'n':
                ret[0] = 1;
                ret[1] = 0;
                ret[2] = -1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'o':
                ret[0] = 1;
                ret[1] = 1;
                ret[2] = 1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'p':
                ret[0] = 0;
                ret[1] = 1;
                ret[2] = 1;
                ret[3] = 0;
                ret[4] = -1;
                break;

            case 'q':
                ret[0] = 1;
                ret[1] = 1;
                ret[2] = 0;
                ret[3] = 1;
                ret[4] = -1;
                break;

            case 'r':
                ret[0] = 0;
                ret[1] = 1;
                ret[2] = 0;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 's':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 't':
                ret[0] = 1;
                ret[1] = -1;
                ret[2] = -1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'u':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = 1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'v':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = 1;
                ret[4] = -1;
                break;

            case 'w':
                ret[0] = 0;
                ret[1] = 1;
                ret[2] = 1;
                ret[3] = -1;
                ret[4] = -1;
                break;

            case 'x':
                ret[0] = 1;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = 1;
                ret[4] = -1;
                break;

            case 'y':
                ret[0] = 1;
                ret[1] = 0;
                ret[2] = 1;
                ret[3] = 1;
                ret[4] = -1;
                break;

            case 'z':
                ret[0] = 1;
                ret[1] = 1;
                ret[2] = 0;
                ret[3] = 0;
                ret[4] = -1;
                break;

            case '1':
                ret[0] = 0;
                ret[1] = 1;
                ret[2] = 1;
                ret[3] = 1;
                ret[4] = 1;
                break;

            case '2':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = 1;
                ret[3] = 1;
                ret[4] = 1;
                break;

            case '3':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = 1;
                ret[4] = 1;
                break;

            case '4':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = 0;
                ret[4] = 1;
                break;

            case '5':
                ret[0] = 0;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = 0;
                ret[4] = 0;
                break;

            case '6':
                ret[0] = 1;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = 0;
                ret[4] = 0;
                break;

            case '7':
                ret[0] = 1;
                ret[1] = 1;
                ret[2] = 0;
                ret[3] = 0;
                ret[4] = 0;
                break;

            case '8':
                ret[0] = 1;
                ret[1] = 1;
                ret[2] = 1;
                ret[3] = 0;
                ret[4] = 0;
                break;

            case '9':
                ret[0] = 1;
                ret[1] = 1;
                ret[2] = 1;
                ret[3] = 1;
                ret[4] = 0;
                break;

            case '0':
                ret[0] = 1;
                ret[1] = 1;
                ret[2] = 1;
                ret[3] = 1;
                ret[4] = 1;
                break;

            case ' ':
                ret[0] = 1;
                ret[1] = 0;
                ret[2] = 0;
                ret[3] = 0;
                ret[4] = 1;
                break;

            default: for(int i = 0; i <5 ; i++) ret[i] = -1;
        }


        return ret;
    }
}
