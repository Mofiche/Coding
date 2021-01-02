package com.example.zeiterfassung;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.TextView;

public class Projekt extends AppCompatActivity {

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if(keyCode == KeyEvent.KEYCODE_BACK) {
            System.out.println("!!!");
            final Projekt projekt = this;
            Intent intent = new Intent(projekt, ProjektLaden.class);
            startActivity(intent);
        }
        return super.onKeyDown(keyCode, event);
    }

    @SuppressLint("ResourceType")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        RelativeLayout rel = new RelativeLayout(this);
        TextView projekt_name = new TextView(this);




        projekt_name.setText(R.string.neues_projekt);
        projekt_name.setId(1);



        RelativeLayout.LayoutParams projekt_name_details = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);



        projekt_name_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        projekt_name_details.addRule(RelativeLayout.BELOW);


        final Projekt main = this;

        projekt_name.setOnClickListener(
                new Button.OnClickListener(){
                    public void onClick(View v){
                        Intent i = new Intent(main, NeuesProjekt.class);
                        startActivity(i);
                    }
                }

        );



        rel.addView(projekt_name,projekt_name_details);


        setContentView(rel);
    }
}