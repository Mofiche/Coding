package CGOL;

import javax.swing.*;
import java.awt.*;

public class Window extends JFrame {


    public static void main(String[] args){
        final int width = 1280;
        final int height = 720;
        Window w = new Window(width, height);
    }

    private Window(int width, int height){
        setLayout(new BorderLayout());
        setSize(width, height);
        setTitle("Conways Game of Life");
        MyCanvas canvas = new MyCanvas();
        add("Center", canvas);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setResizable(false);
        setVisible(true);
    }

}


