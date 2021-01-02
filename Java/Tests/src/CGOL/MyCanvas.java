package CGOL;

import java.awt.*;
import java.util.Random;

class MyCanvas extends Canvas {

    @Override
    public void paint(Graphics g){
        Random rand = new Random();
        int width = getWidth();
        int height = getHeight();
        int res = 2;
        int cols = width / res;
        int rows = height/ res;

        boolean[][] grid = new boolean[cols][rows];
        for(int i = 0 ;i < cols;i++){
            for(int j = 0 ; j< rows ;j++){
                grid[i][j] = rand.nextBoolean();
            }
        }

        while(true){
            for(int i = 0 ;i < cols;i++){
                for(int j = 0 ; j< rows ;j++){
                    if(grid[i][j]){
                        g.setColor(Color.BLACK);
                    }else{
                        g.setColor(Color.WHITE);
                    }
                    g.fillRect(i*res, j*res, res,res);
                }
            }

            boolean[][] next = new boolean[cols][rows];

            for(int i = 0 ;i < cols;i++) {
                for (int j = 0; j < rows; j++) {
                    boolean state = grid[i][j];

                    int neighbors =0;
                    for(int x = -1 ; x <= 1; x++){
                        for(int y = -1 ; y <= 1 ; y++){

                            int col = (x + i + cols)%cols;
                            int row = (y + j + rows)%rows;

                            if(grid[col][row]) neighbors+=1;
                        }
                    }
                    if(grid[i][j]) neighbors -=1;

                    if(!state && neighbors == 3){
                        next[i][j] = true;
                    } else if(state && (neighbors <2 || neighbors > 3)){
                        next[i][j]= false;
                    }else{
                        next[i][j] = state;
                    }
                }
            }
            grid = next;
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }


    }

    private int  count(boolean[][] grid,int x,int y){
        int sum = 0 ;
        for(int i = -1 ; i < 2; i++){
            for(int j = -1 ; j < 2 ; j++){
                if(grid[x+i][y+j]) sum+=1;
            }
        }
        if(grid[x][y]) sum-= 1;
        return sum;
    }

}