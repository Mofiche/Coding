package AStar;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;

import static AStar.Utils.heuristic;

public class MyCanvas extends Canvas {

    private void freeze(){
        try {
            Thread.sleep(100 * 60 * 60 * 24 );
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void paint(Graphics g) {
        final int width = Window.width;
        final int height = Window.height;
        final int cols = Window.cols;
        final int rows = Window.rows;
        final int w = width / cols;
        final int h = height / rows;
        Spot[] path = new Spot[0];
        Spot[][] grid = new Spot[cols][rows];
        Spot[] openSet = new Spot[0];
        Spot[] closedSet = new Spot[0];
        Spot start, end;

        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                grid[i][j] = new Spot(i, j, w, h, cols, rows);
            }
        }

        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                grid[i][j].addNeighbors(grid);
            }
        }

        start = grid[0][0];
        start.wall = false;
        end = grid[cols - 1][rows - 1];
        end.wall = false;
        openSet = Utils.add(openSet, start);

        while (openSet.length > 0 ) {
            int winner = 0;

            for (int i = 0; i < openSet.length; i++) {
                if (openSet[i].f < openSet[winner].f) winner = i;
            }


            Spot current = openSet[winner];

            if (current == end) {
                Spot temp = current;
                path = Utils.add(path, temp);
                while (!(temp.pre == null)) {
                    path = Utils.add(path, temp.pre);
                    temp = temp.pre;
                }
                Window.length = path.length;
                System.out.println("Fertig");
                System.out.println("Länge des Weges " + path.length + " Längeneinheiten");
            }

            openSet = Utils.remove(openSet, current);
            closedSet = Utils.add(closedSet, current);

            Spot[] neighbors = current.neighbors;
            for (Spot neighbor : neighbors) {
                if (!(Arrays.asList(closedSet).contains(neighbor)) && !neighbor.wall) {
                    double tempg = current.g + 1;
                    boolean newPath  = false;
                    if (Arrays.asList(openSet).contains(neighbor)) {
                        if (tempg < neighbor.g) {
                            neighbor.g = tempg;
                            newPath = true;
                        }
                    } else {
                        neighbor.g = tempg;
                        newPath = true;
                        openSet = Utils.add(openSet, neighbor);
                    }
                    if(newPath){
                        neighbor.h = heuristic(neighbor, end);
                        neighbor.f = (neighbor.g + neighbor.h);
                        neighbor.pre = current;
                    }
                }

            }
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    boolean conopen = Arrays.asList(openSet).contains(grid[i][j]);
                    boolean conclose = Arrays.asList(closedSet).contains(grid[i][j]);
                    if (!conopen && !conclose)
                        grid[i][j].show(g, Color.WHITE);
                }
            }

            if(path.length == 0 ){
                for (Spot anOpenSet : openSet) {
                    anOpenSet.show(g, Color.GREEN);
                    try {
                        Thread.sleep(0);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }

            if ((path.length > 0)) {
                for (int i = 0; i < cols; i++) {
                    for (int j = 0; j < rows; j++) {
                        grid[i][j].show(g, Color.WHITE);
                    }
                }

                for (Spot aPath : path) {
                    aPath.show(g, Color.BLUE);
                }

                for(int i = 0 ; i < openSet.length; i++){
                    openSet = Utils.remove(openSet,openSet[i]);
                }
                freeze();
            }
        }
    }
}
