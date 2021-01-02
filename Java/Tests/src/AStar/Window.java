package AStar;

import javax.swing.*;
import java.awt.*;

public class Window extends JFrame {
    static final int cols = 250;
    static final int rows = 250;
    static final double densityWall  = 0.4;
    static final int width = 1000;
    static final int height = 1000;
    static int length = 0 ;


    public static void main(String[] args){
        final int width = ((Window.width / cols) * cols) +5;
        final int height = ((Window.height/ rows ) * rows) + 30;
        Window w = new Window(width, height);
    }

    private Window(int width, int height){
        setLayout(new BorderLayout());
        setSize(width, height);
        setTitle("AStar - Pathfinding");
        MyCanvas canvas = new MyCanvas();
        add("Center", canvas);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setResizable(false);
        setVisible(true);
    }

}


