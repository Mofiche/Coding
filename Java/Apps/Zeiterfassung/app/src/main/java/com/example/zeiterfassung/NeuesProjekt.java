package com.example.zeiterfassung;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RelativeLayout;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Date;

public class NeuesProjekt extends AppCompatActivity {

    @SuppressLint("SetTextI18n")
    boolean projekt_initialisieren(String name, Button btn){
        File directory = getFilesDir();
        System.out.println(directory);
        File file = new File(directory,name.concat(".dat"));
        System.out.println(file);
        try {
            if(!file.exists() && !name.isEmpty()) {
                FileOutputStream fileStream = new FileOutputStream(file);
                ObjectOutputStream objectStream = new ObjectOutputStream(fileStream);

                objectStream.writeObject(name);
                objectStream.writeObject(new Date().toString().toCharArray());
                objectStream.writeObject(0);

                objectStream.close();
                fileStream.close();
                return true;
            }else{
                btn.setTextColor(Color.RED);
                if(name.isEmpty()){
                    btn.setText("Bitte einen Name eingeben!");
                }else {
                    btn.setText("Dieses Projekt gibt es schon!");
                }
                return false;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if(keyCode == KeyEvent.KEYCODE_BACK) {
            System.out.println("!!!");
            final NeuesProjekt projekt = this;
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
        final EditText projekt_name = new EditText(this);
        final Button projekt_speichern = new Button(this);




        projekt_name.setText(R.string.NeuesProjekt_NameProjekt);
        projekt_name.setId(1);
        projekt_name.setText("");
        projekt_name.setWidth(500);
        projekt_name.setCursorVisible(true);
        projekt_speichern.setText(R.string.neues_projekt_fenster_bestaetigung);
        projekt_speichern.setId(2);


        RelativeLayout.LayoutParams projekt_name_details = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        RelativeLayout.LayoutParams projekt_speichern_details = new RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);


        projekt_name_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        projekt_name_details.addRule(RelativeLayout.BELOW);
        projekt_speichern_details.addRule(RelativeLayout.CENTER_HORIZONTAL);
        projekt_speichern_details.addRule(RelativeLayout.BELOW,1);

        final NeuesProjekt projekt = this;

        projekt_speichern.setOnClickListener(
                new Button.OnClickListener(){
                    public void onClick(View v){
                        String name = projekt_name.getText().toString();
                        boolean b = projekt_initialisieren(name, projekt_speichern);
                        if(b) {
                            Intent intent = new Intent(projekt, MainActivity.class);
                            startActivity(intent);
                        }
                    }
                }

        );



        rel.addView(projekt_name,projekt_name_details);
        rel.addView(projekt_speichern,projekt_speichern_details);


        setContentView(rel);
    }
}