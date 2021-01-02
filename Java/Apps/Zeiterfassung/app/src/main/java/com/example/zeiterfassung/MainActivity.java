package com.example.zeiterfassung;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    // ID´s
    /*

    1: Neues Projekt : Button
    2: Projekt Laden : Button
    3: Tageszusammenfassung : Button
    4: Projektzusammenfassung : Button

     */

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if(keyCode == KeyEvent.KEYCODE_BACK) {
            System.exit(0);
        }
        return super.onKeyDown(keyCode, event);
    }

    @SuppressLint("ResourceType")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        RelativeLayout rel = new RelativeLayout(this);
        final Button neues_projekt = new Button(this);
        Button projekt_laden = new Button(this);
        Button tageszusammenfassung = new Button(this);
        Button projektzusammenfassung = new Button(this);
        Button projektloeschen = new Button(this);



        neues_projekt.setText(R.string.neues_projekt);
        neues_projekt.setId(1);
        projekt_laden.setText(R.string.projekt_laden);
        projekt_laden.setId(2);
        tageszusammenfassung.setText(R.string.tageszusammenfassung);
        tageszusammenfassung.setId(3);
        projektzusammenfassung.setText(R.string.projektzusammenfassung);
        projektzusammenfassung.setId(4);
        projektloeschen.setText(R.string.projekt_loeschen);


        RelativeLayout.LayoutParams neues_projekt_details = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        RelativeLayout.LayoutParams projekt_laden_details = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        RelativeLayout.LayoutParams tageszusammenfassung_details = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        RelativeLayout.LayoutParams projektzusammenfassung_details = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        RelativeLayout.LayoutParams projektloeschen_details = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);


        neues_projekt_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        neues_projekt_details.addRule(RelativeLayout.BELOW);
        projekt_laden_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        projekt_laden_details.addRule(RelativeLayout.BELOW,1);
        tageszusammenfassung_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        tageszusammenfassung_details.addRule(RelativeLayout.BELOW,2);
        projektzusammenfassung_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        projektzusammenfassung_details.addRule(RelativeLayout.BELOW,3);
        projektloeschen_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        projektloeschen_details.addRule(RelativeLayout.BELOW,4);

        final MainActivity main = this;

        neues_projekt.setOnClickListener(
                new Button.OnClickListener(){
                    public void onClick(View v){
                        Intent i = new Intent(main, NeuesProjekt.class);
                        startActivity(i);
                    }
                }

        );

        projekt_laden.setOnClickListener(
                new Button.OnClickListener(){
                    public void onClick(View v){
                        Intent i = new Intent(main, ProjektLaden.class);
                        startActivity(i);
                    }
                }

        );

        tageszusammenfassung.setOnClickListener(
                new Button.OnClickListener(){
                    public void onClick(View v){
                        //
                    }
                }

        );

        projektzusammenfassung.setOnClickListener(
                new Button.OnClickListener(){
                    public void onClick(View v){
                        Intent i = new Intent(main, Projektzusammenfassung.class);
                        startActivity(i);
                    }
                }

        );

        projektloeschen.setOnClickListener(
                new Button.OnClickListener(){
                    public void onClick(View v){
                        Intent i = new Intent(main, ProjektLoeschen.class);
                        startActivity(i);
                    }
                }

        );

        rel.addView(projekt_laden,projekt_laden_details);
        rel.addView(neues_projekt,neues_projekt_details);
        rel.addView(tageszusammenfassung,tageszusammenfassung_details);
        rel.addView(projektzusammenfassung,projektzusammenfassung_details);
        rel.addView(projektloeschen,projektloeschen_details);

        setContentView(rel);
    }

    @Override
    protected void onStart() {
        super.onStart();
    }

    @Override
    protected void onStop() {
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }


}