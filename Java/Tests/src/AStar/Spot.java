package AStar;

import java.awt.*;
import java.util.Random;

class Spot {

    int i , j;
    double f = 0;
    double g = 0;
    double h = 0;
    private int w, he;
    private int cols, rows;
    Spot[] neighbors = new Spot[0];
    Spot pre = null;
    boolean wall = false;
    private Random rand = new Random();

    Spot(int i, int j , int w , int h, int cols, int rows){
        this.i = i;
        this.j = j;
        this.w = w;
        this.he = h;
        this.cols = cols;
        this.rows = rows;
        if(rand.nextDouble() < Window.densityWall){
            this.wall = true;
        }
    }

    void show(Graphics g, Color col ){
        g.setColor(col);
        if(this.wall){
            g.setColor(Color.BLACK);
        }
        g.fillRect(i*w,j*he,w,he);
        g.setColor(Color.BLACK);
        g.drawRect(i * w, j*he , w , he);
    }

    void addNeighbors( Spot[][] grid ){
        if( i < cols - 1 ) neighbors = Utils.add(neighbors,grid[i+1][j]);
        if( i > 0 ) neighbors = Utils.add(neighbors,grid[i-1][j]);
        if( j < rows - 1 ) neighbors = Utils.add(neighbors,grid[i][j+1]);
        if( j > 0 ) neighbors = Utils.add(neighbors,grid[i][j-1]);
        if(i > 0 &&  j > 0 ) neighbors = Utils.add(neighbors,grid[i-1][j-1]);
        if(i < cols -1 && j > 0 ) neighbors = Utils.add(neighbors,grid[i+1][j-1]);
        if(i > 0 && j < rows -1 ) neighbors = Utils.add(neighbors,grid[i-1][j+1]);
        if(i < cols -1 && j < rows -1 ) neighbors = Utils.add(neighbors,grid[i+1][j+1]);




    }


}
