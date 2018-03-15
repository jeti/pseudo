package io.pcess.trajectory_optimization;

import static android.os.SystemClock.elapsedRealtimeNanos;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        TextView tv = (TextView) findViewById(R.id.sample_text);
        double t_start = (double) elapsedRealtimeNanos()/1e9;
        double t_opt = optimizetrajectory();
        tv.setText("Optimal time = " + t_opt + " secs");
        double t_end = (double) elapsedRealtimeNanos()/1e9;
        double t = t_end - t_start;
        TextView tv_time = (TextView) findViewById(R.id.run_time);
        tv_time.setText("Run time: " + t + " secs");
    }

    /**
     * A native method that is implemented by the native library,
     * which is packaged with this application.
     */
    public native double optimizetrajectory();
}
