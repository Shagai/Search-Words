package org.opencv.samples.tutorial3;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;

import org.opencv.tracking.Tracker;
import org.opencv.video.Video;

public class MainActivity extends Activity implements CvCameraViewListener2, OnTouchListener {

    static {
        System.loadLibrary("opencv_java3");
    }

    private static final String TAG = "OCVSample::Activity";

    private MainView mOpenCvCameraView;
    private List<Size> mResolutionList;
    private MenuItem[] mEffectMenuItems;
    private SubMenu mColorEffectsMenu;
    private MenuItem[] mResolutionMenuItems;
    private SubMenu mResolutionMenu;

    private DetectMarks     det = new DetectMarks();
    private boolean         task = false;
    private List<MatOfPoint> squares = null;
    private int state = 0;
    private Mat prevImg = null;
    private MatOfPoint2f prevFeatures = new MatOfPoint2f();


    // Tracker OpenCV
    Mat descriptor = null;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {Log.i(TAG, "Instantiated new " + this.getClass());}

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.tutorial3_surface_view);

        mOpenCvCameraView = (MainView) findViewById(R.id.tutorial3_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        Mat rgba = inputFrame.rgba();
        final Mat gray = inputFrame.gray();
        org.opencv.core.Size sizeRgba = rgba.size();

        Mat rgbaInnerWindow;

        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;

        /*
        if (this.task == false && (this.state != 1)) {
            det.execute(rgba, gray);
            this.task = true;
        }

        List<MatOfPoint> squares = det.GetSquares();

        if (!det.isBusy() && (this.state != 1)) {
            //rgba = frame.clone(); // Desremove //  when you want fluid video
            det = new DetectMarks();
            this.task = false;
            this.squares = squares;
        }
        */

        if(this.state == 1){
            MatOfPoint2f newFeatures;
            newFeatures = FeatureTracking(gray);
            this.squares = new ArrayList<>();
            MatOfPoint quad = new MatOfPoint();
            newFeatures.convertTo(quad, CvType.CV_32S);
            this.squares.add(quad);
            Imgproc.drawContours(rgba, this.squares, -1, new Scalar(0, 0, 255), 2);
        }

        if(this.state == 0){
            this.squares = new ArrayList<>();
            if(Find_Squares(rgba, this.squares) == 1) {
                this.prevImg = gray.clone();
                this.squares.get(0).convertTo(this.prevFeatures, CvType.CV_32FC3);
                Imgproc.drawContours(rgba, this.squares, -1, new Scalar(0, 0, 255), 2);
                this.state = 1;
            }
        }

        return rgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {

        return true;
    }

    public boolean onOptionsItemSelected(MenuItem item) {

        return true;
    }

    @SuppressLint("SimpleDateFormat")
    @Override
    public boolean onTouch(View v, MotionEvent event) {
        mOpenCvCameraView.setupCameraFlashLight();
        return false;
    }

    private Rect ConvertMoP2Rect(MatOfPoint square){
        Point vertex1 = new Point(square.get(0,0)[0], square.get(0,0)[1]);
        Point vertex2 = new Point(square.get(2,0)[0], square.get(2,0)[1]);

        return new Rect(vertex1, vertex2);
    }

    private MatOfPoint Center(MatOfPoint square){
        Point mid = new Point();
        mid.x = (square.get(0,0)[0] + square.get(2,0)[0]) / 2;
        mid.y = (square.get(0,0)[1] + square.get(2,0)[1]) / 2;
        return new MatOfPoint(mid);
    }

    private int Find_Squares(Mat img, List<MatOfPoint> quads) {
        Mat bin = new Mat();
        Mat cannyDst = new Mat();

        Imgproc.Canny(img, cannyDst, 0, 50);
        Mat dilated_ker = Mat.ones(1, 1, CvType.CV_32F);
        //Imgproc.dilate(cannyDst, bin, dilated_ker);
        List<MatOfPoint> contours = new ArrayList<>();
        //quads = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(cannyDst, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint cnt : contours) {
            MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
            double cnt_len = Imgproc.arcLength(cnt2f, true);
            Imgproc.approxPolyDP(cnt2f, cnt2f, 0.02 * cnt_len, true);
            cnt2f.convertTo(cnt, CvType.CV_32S);
            if (cnt2f.height() == 4 && Imgproc.contourArea(cnt2f) > 10000){

                double[] cos = new double[4];

                for (int i = 0; i < 4; i++){
                    double[] p0 = cnt.get(i, 0);
                    double[] p1 = cnt.get((i + 1)%4, 0);
                    double[] p2 = cnt.get((i + 2)%4, 0);
                    cos[i] = angle_cos(p0, p1, p2);
                }

                if (getMax(cos) < 0.3){
                    quads.add(cnt);
                    return 1;
                }

            }
        }

        return -1;
    }

    private double angle_cos(double[] p0, double[] p1, double[] p2){
        double[] d1 = new double[2];
        double[] d2 = new double[2];

        for (int i = 0; i < 2; i++) {
            d1[i] = p0[i] - p1[i];
            d2[i] = p2[i] - p1[i];
        }

        double dot = d1[0] * d2[0] + d1[1] * d2[1];
        double dot1 = d1[0] * d1[0] + d1[1] * d1[1];
        double dot2 = d2[0] * d2[0] + d2[1] * d2[1];

        return Math.abs(dot / Math.sqrt(dot1 * dot2));
    }

    private static double getMax(double[] array){
        double max = 0;
        for (double num : array) {
            if (num > max) {
                max = num;
            }
        }

        return max;
    }

    private void StartTracking(Mat image, List<MatOfPoint> squares){

    }

    private MatOfPoint2f FeatureTracking(Mat newImage){
        MatOfPoint2f newFeatures = new MatOfPoint2f();
        MatOfByte status = new MatOfByte();
        MatOfFloat err = new MatOfFloat();
        Video.calcOpticalFlowPyrLK(this.prevImg, newImage, this.prevFeatures, newFeatures, status, err);
        this.prevFeatures = newFeatures;
        this.prevImg = newImage.clone();
        return newFeatures;
    }
}
