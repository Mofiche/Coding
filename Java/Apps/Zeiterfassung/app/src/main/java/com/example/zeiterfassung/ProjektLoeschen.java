package com.example.zeiterfassung;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.ScrollView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;

public class ProjektLoeschen extends AppCompatActivity {

    String formatieren(String string){
        return string.replaceAll(".dat","");
    }

    Boolean abfrage(final String name){
        final Boolean[] ret = {false};
        AlertDialog.Builder dialog = new AlertDialog.Builder(this);
        dialog.setMessage("Wollen sie das Projekt '" + name + "' wirklich löschen?");
        dialog.setTitle("");
        dialog.setCancelable(true);
        dialog.setPositiveButton("Ja", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                @SuppressLint("ShowToast") Toast toast = Toast.makeText(getApplicationContext(), "Projekt ".concat(name).concat(" wurde gelöscht!"),Toast.LENGTH_LONG);
                ret[0] = true;
            }
        });
        dialog.setNegativeButton("Nein", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                ret[0] = false;
            }
        });
        AlertDialog alert = dialog.create();
        alert.show();
        return ret[0];
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if(keyCode == KeyEvent.KEYCODE_BACK) {
            System.out.println("!!!");
            final ProjektLoeschen projekt = this;
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
        final ProjektLoeschen projekt = this;
        int length = 0;
        if (files != null) {
            length = files.length;
        }
        final Button[] buttons = new Button[length];
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

        for(int i = 0; i<= buttons.length - 1;i++) {
            buttons[i] = new Button(this);
            buttons[i].setText(formatieren(files[i]));
            buttons[i].setId(i + 1);
            button_details[i] = new RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
            button_details[i].addRule(RelativeLayout.CENTER_HORIZONTAL);
            if (i != 0) {
                button_details[i].addRule(RelativeLayout.BELOW, i);
            } else {
                button_details[i].addRule(RelativeLayout.BELOW);
            }
            final int j = i;
            buttons[i].setOnClickListener(
                    new Button.OnClickListener() {
                        @Override
                        public void onClick(View view) {
                            File directory = getFilesDir();
                            File file = new File(directory, ((String) buttons[j].getText()).concat(".dat"));
                            boolean del = false;
                            if(abfrage((String) buttons[j].getText())) {
                                del = file.delete();
                            }
                            if (!del) {
                                System.out.println("Datei konnte nicht gelöscht werden : ".concat((String) buttons[j].getText()).concat(".dat"));
                            }
                            Intent intent = new Intent(projekt, ProjektLoeschen.class);
                            startActivity(intent);
                        }
                    }
            );
            layout.addView(buttons[i], button_details[i]);

        }
        scrollview.addView(layout,layout_details);
        rel.addView(scrollview,scrollview_details);

        setContentView(rel);
    }
}