package com.example.zeiterfassung;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.ScrollView;

import java.io.File;

public class Projektzusammenfassung extends AppCompatActivity {

    String formatieren(String string){
        return string.replaceAll(".dat","");
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if(keyCode == KeyEvent.KEYCODE_BACK) {
            System.out.println("!!!");
            final Projektzusammenfassung projekt = this;
            Intent intent = new Intent(projekt, MainActivity.class);
            startActivity(intent);
        }
        return super.onKeyDown(keyCode, event);
    }

    @SuppressLint("ResourceType")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        RelativeLayout rel = new RelativeLayout(this);
        File directory = getFilesDir();
        String[] files = directory.list();
        final Projektzusammenfassung projekt = this;
        final Button[] buttons;
        int length = 0;
        if (files != null) {
            length = files.length;
        }
        buttons = new Button[length];

        final RelativeLayout.LayoutParams[] button_details = new RelativeLayout.LayoutParams[buttons.length];
        ScrollView scrollview = new ScrollView(this);
        RelativeLayout layout = new RelativeLayout(this);


        scrollview.setId(-1);
        layout.setId(-2);
        RelativeLayout.LayoutParams scrollview_details = new RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        RelativeLayout.LayoutParams layout_details = new RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);

        scrollview_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        layout_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        scrollview_details.addRule(RelativeLayout.BELOW);
        layout_details.addRule(RelativeLayout.BELOW);

        if (files != null) {
            for(int i = 0; i<=files.length - 1;i++){
                System.out.println(files[i]);
            }
        }

        for(int i = 0; i<= buttons.length - 1;i++){
            buttons[i] = new Button(this);
            buttons[i].setText(formatieren(files[i]));
            buttons[i].setId(i+1);
            button_details[i] = new RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
            button_details[i].addRule(RelativeLayout.CENTER_HORIZONTAL);
            if(i != 0) {
                button_details[i].addRule(RelativeLayout.BELOW, i);
            }else {
                button_details[i].addRule(RelativeLayout.BELOW,-1);
            }
            buttons[i].setOnClickListener(
                    new Button.OnClickListener(){
                        @Override
                        public void onClick(View view) {
                            Intent intent = new Intent(projekt, Projekt.class);
                            startActivity(intent);

                        }
                    }
            );

            layout.addView(buttons[i],button_details[i]);


        }
        scrollview.addView(layout,layout_details);
        rel.addView(scrollview,scrollview_details);
        setContentView(rel);
    }
}