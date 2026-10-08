package com.syncpods.app;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.media.AudioAttributes;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {
    private static final int REQ_BT = 7001;
    private static final int REQ_OPEN_AUDIO = 7003;

    private AudioManager audioManager;
    private MediaPlayer mediaPlayer;
    private AudioTrack demoTrack;
    private LinearLayout outputsContainer;
    private TextView statusText, countText;
    private Button playButton;
    private Uri selectedAudio;

    private int dp(float v) {
        return (int)(v * getResources().getDisplayMetrics().density + 0.5f);
    }

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(Color.rgb(11,15,20));
        getWindow().setNavigationBarColor(Color.rgb(11,15,20));
        audioManager = (AudioManager)getSystemService(Context.AUDIO_SERVICE);
        buildUi();
        ensurePermissions();
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(18), dp(20), dp(28));
        root.setBackgroundColor(Color.rgb(11,15,20));
        scroll.addView(root);

        TextView brand = tv("SYNCPODS", 13, Color.rgb(0,212,255), true);
        brand.setLetterSpacing(0.18f);
        root.addView(brand, lp(-1,-2,0,0,dp(4),0));

        TextView title = tv("One song.\nMultiple Pods.", 32, Color.WHITE, true);
        root.addView(title, lp(-1,-2,0,dp(12),0,0));

        TextView sub = tv("Play the same audio through the Bluetooth outputs selected by your phone.", 15, Color.rgb(147,164,183), false);
        sub.setLineSpacing(0,1.15f);
        root.addView(sub, lp(-1,-2,0,0,0,0));

        LinearLayout hero = card();
        root.addView(hero, lp(-1,-2,0,dp(18),0,0));
        hero.addView(tv("AUDIO ROUTING",12,Color.rgb(0,212,255),true), lp(-1,-2,0,0,0,dp(8)));
        countText = tv("Checking outputs…",22,Color.WHITE,true);
        hero.addView(countText, lp(-1,-2,0,0,0,dp(4)));
        statusText = tv("Connect your AirPods in Bluetooth settings.",13,Color.rgb(147,164,183),false);
        hero.addView(statusText, lp(-1,-2,0,0,0,0));

        outputsContainer = new LinearLayout(this);
        outputsContainer.setOrientation(LinearLayout.VERTICAL);
        root.addView(outputsContainer, lp(-1,-2,0,dp(12),0,0));

        LinearLayout controls = card();
        root.addView(controls, lp(-1,-2,0,dp(6),0,0));
        controls.addView(tv("PLAYBACK",12,Color.rgb(0,212,255),true), lp(-1,-2,0,0,0,dp(10)));

        Button choose = button("Choose song");
        choose.setOnClickListener(v -> chooseAudio());
        controls.addView(choose, lp(-1,dp(52),0,0,0,dp(10)));

        playButton = button("▶  Play on selected outputs");
        playButton.setOnClickListener(v -> togglePlayback());
        controls.addView(playButton, lp(-1,dp(54),0,0,0,dp(10)));

        Button refresh = secondaryButton("↻  Refresh connected outputs");
        refresh.setOnClickListener(v -> refreshStatus());
        controls.addView(refresh, lp(-1,dp(48),0,0,0,0));

        Button bluetooth = secondaryButton("Bluetooth settings");
        bluetooth.setOnClickListener(v -> {
            try { startActivity(new Intent(Settings.ACTION_BLUETOOTH_SETTINGS)); }
            catch (Exception e) { startActivity(new Intent(Settings.ACTION_SETTINGS)); }
        });
        root.addView(bluetooth, lp(-1,dp(48),0,dp(10),0,0));

        TextView note = tv("Important: SyncPods does not bypass Android audio-routing rules.\nOn phones with Dual Audio / LE Audio sharing, enable multiple outputs in the system audio panel first.",12,Color.rgb(147,164,183),false);
        note.setLineSpacing(0,1.25f);
        root.addView(note, lp(-1,-2,0,dp(16),0,0));

        setContentView(scroll);
    }

    private void ensurePermissions() {
        List<String> needed = new ArrayList<>();
        if (Build.VERSION.SDK_INT >= 31) {
            if (checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) needed.add(Manifest.permission.BLUETOOTH_CONNECT);
            if (checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED) needed.add(Manifest.permission.BLUETOOTH_SCAN);
        }
        if (!needed.isEmpty()) requestPermissions(needed.toArray(new String[0]), REQ_BT);
        else refreshStatus();
    }

    private void refreshStatus() {
        outputsContainer.removeAllViews();
        List<AudioDeviceInfo> outputs = new ArrayList<>();
        try {
            for (AudioDeviceInfo d : audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)) {
                int t = d.getType();
                if (t == AudioDeviceInfo.TYPE_BLUETOOTH_A2DP ||
                    t == AudioDeviceInfo.TYPE_BLUETOOTH_SCO ||
                    (Build.VERSION.SDK_INT >= 31 && (t == AudioDeviceInfo.TYPE_BLE_HEADSET || t == AudioDeviceInfo.TYPE_BLE_SPEAKER || t == AudioDeviceInfo.TYPE_BLE_BROADCAST))) {
                    outputs.add(d);
                }
            }
        } catch (SecurityException ignored) {}

        countText.setText(outputs.size() + " Bluetooth audio output" + (outputs.size() == 1 ? "" : "s") + " detected");
        statusText.setText(outputs.size() >= 2
                ? "Multiple outputs are visible. System multi-output support may be available."
                : "Pair and connect your AirPods, then refresh.");

        if (outputs.isEmpty()) {
            outputsContainer.addView(deviceCard("No connected Bluetooth audio output","Connect AirPods first",false), lp(-1,-2,0,0,0,dp(8)));
        } else {
            for (int i=0; i<outputs.size() && i<5; i++) {
                AudioDeviceInfo d = outputs.get(i);
                String name = d.getProductName() == null ? "Bluetooth audio" : d.getProductName().toString();
                outputsContainer.addView(deviceCard(name, readableType(d.getType()) + " • Output #" + (i+1), true), lp(-1,-2,0,0,0,dp(8)));
            }
        }
    }

    private View deviceCard(String name, String subtitle, boolean active) {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.HORIZONTAL);
        box.setGravity(Gravity.CENTER_VERTICAL);
        box.setPadding(dp(16),dp(14),dp(16),dp(14));
        android.graphics.drawable.GradientDrawable bg = new android.graphics.drawable.GradientDrawable();
        bg.setColor(Color.rgb(18,25,35)); bg.setCornerRadius(dp(16));
        box.setBackground(bg);

        TextView icon = tv(active ? "●" : "○",18,active ? Color.rgb(36,209,126) : Color.rgb(147,164,183),true);
        box.addView(icon, lp(dp(32),-2,0,0,0,0));

        LinearLayout texts = new LinearLayout(this);
        texts.setOrientation(LinearLayout.VERTICAL);
        texts.addView(tv(name,15,Color.WHITE,true),lp(-1,-2,0,0,0,dp(3)));
        texts.addView(tv(subtitle,12,Color.rgb(147,164,183),false),lp(-1,-2,0,0,0,0));
        box.addView(texts,lp(0,-2,0,0,0,0,1f));
        return box;
    }

    private void togglePlayback() {
        if (mediaPlayer != null && mediaPlayer.isPlaying()) {
            mediaPlayer.pause();
            playButton.setText("▶  Play on selected outputs");
            return;
        }
        if (demoTrack != null && demoTrack.getPlayState() == AudioTrack.PLAYSTATE_PLAYING) {
            demoTrack.pause();
            playButton.setText("▶  Play on selected outputs");
            return;
        }
        startPlayback();
    }

    private void startPlayback() {
        stopPlayer();
        try {
            if (selectedAudio != null) {
                mediaPlayer = new MediaPlayer();
                mediaPlayer.setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build());
                mediaPlayer.setDataSource(this, selectedAudio);
                mediaPlayer.setOnPreparedListener(mp -> {
                    mp.start();
                    playButton.setText("⏸  Pause");
                    Toast.makeText(this,"Playback started through Android audio routing.",Toast.LENGTH_LONG).show();
                });
                mediaPlayer.setOnCompletionListener(mp -> playButton.setText("▶  Play on selected outputs"));
                mediaPlayer.prepareAsync();
            } else {
                playDemoTone();
            }
        } catch (Exception e) {
            Toast.makeText(this,"Playback failed: " + e.getMessage(),Toast.LENGTH_LONG).show();
        }
    }

    private void playDemoTone() {
        final int sampleRate = 44100;
        final int durationMs = 3000;
        final int count = sampleRate * durationMs / 1000;
        short[] samples = new short[count];
        for (int i=0;i<count;i++) {
            double env = 1.0 - (double)i / count;
            samples[i] = (short)(Math.sin(2.0*Math.PI*440*i/sampleRate) * 12000 * env);
        }

        demoTrack = new AudioTrack(
                new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build(),
                new AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT).setSampleRate(sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build(),
                samples.length * 2, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE);

        demoTrack.write(samples,0,samples.length);
        demoTrack.setNotificationMarkerPosition(samples.length);
        demoTrack.setPlaybackPositionUpdateListener(new AudioTrack.OnPlaybackPositionUpdateListener() {
            @Override public void onMarkerReached(AudioTrack track) { playButton.setText("▶  Play on selected outputs"); }
            @Override public void onPeriodicNotification(AudioTrack track) {}
        });
        demoTrack.play();
        playButton.setText("⏸  Pause");
        Toast.makeText(this,"Demo tone started. Android controls the audio output.",Toast.LENGTH_LONG).show();
    }

    private void stopPlayer() {
        if (mediaPlayer != null) {
            try { mediaPlayer.stop(); } catch (Exception ignored) {}
            mediaPlayer.release();
            mediaPlayer = null;
        }
        if (demoTrack != null) {
            try { demoTrack.stop(); } catch (Exception ignored) {}
            demoTrack.release();
            demoTrack = null;
        }
    }

    private void chooseAudio() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.setType("audio/*");
        i.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(i, REQ_OPEN_AUDIO);
    }

    @Override protected void onActivityResult(int requestCode,int resultCode,Intent data) {
        super.onActivityResult(requestCode,resultCode,data);
        if (requestCode == REQ_OPEN_AUDIO && resultCode == RESULT_OK && data != null && data.getData() != null) {
            selectedAudio = data.getData();
            Toast.makeText(this,"Song selected",Toast.LENGTH_SHORT).show();
        }
    }

    @Override protected void onResume() {
        super.onResume();
        if (outputsContainer != null) refreshStatus();
    }

    @Override protected void onDestroy() {
        stopPlayer();
        super.onDestroy();
    }

    private LinearLayout card() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(16),dp(16),dp(16),dp(16));
        android.graphics.drawable.GradientDrawable bg = new android.graphics.drawable.GradientDrawable();
        bg.setColor(Color.rgb(18,25,35)); bg.setCornerRadius(dp(18));
        box.setBackground(bg);
        return box;
    }

    private TextView tv(String text,float size,int color,boolean bold) {
        TextView t = new TextView(this);
        t.setText(text); t.setTextSize(size); t.setTextColor(color);
        t.setTypeface(Typeface.create("sans",bold ? Typeface.BOLD : Typeface.NORMAL));
        return t;
    }

    private Button button(String text) {
        Button b = new Button(this);
        b.setText(text); b.setTextColor(Color.WHITE); b.setTextSize(14); b.setAllCaps(false);
        android.graphics.drawable.GradientDrawable bg = new android.graphics.drawable.GradientDrawable();
        bg.setColor(Color.rgb(108,99,255)); bg.setCornerRadius(dp(14));
        b.setBackground(bg);
        return b;
    }

    private Button secondaryButton(String text) {
        Button b = new Button(this);
        b.setText(text); b.setTextColor(Color.WHITE); b.setTextSize(13); b.setAllCaps(false);
        android.graphics.drawable.GradientDrawable bg = new android.graphics.drawable.GradientDrawable();
        bg.setColor(Color.rgb(23,33,45)); bg.setCornerRadius(dp(14));
        b.setBackground(bg);
        return b;
    }

    private String readableType(int type) {
        switch(type) {
            case AudioDeviceInfo.TYPE_BLUETOOTH_A2DP: return "Bluetooth A2DP";
            case AudioDeviceInfo.TYPE_BLUETOOTH_SCO: return "Bluetooth SCO";
            case AudioDeviceInfo.TYPE_BLE_HEADSET: return "LE Audio headset";
            case AudioDeviceInfo.TYPE_BLE_SPEAKER: return "LE Audio speaker";
            case AudioDeviceInfo.TYPE_BLE_BROADCAST: return "LE Audio broadcast";
            default: return "Bluetooth audio";
        }
    }

    private LinearLayout.LayoutParams lp(int w,int h,int l,int t,int r,int b) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(w,h);
        p.setMargins(l,t,r,b);
        return p;
    }

    private LinearLayout.LayoutParams lp(int w,int h,int l,int t,int r,int b,float weight) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(w,h,weight);
        p.setMargins(l,t,r,b);
        return p;
    }
}