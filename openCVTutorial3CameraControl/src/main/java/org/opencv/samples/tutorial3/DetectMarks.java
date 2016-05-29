package org.opencv.samples.tutorial3;

import android.os.AsyncTask;

import org.opencv.core.Core;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Jcama on 29/12/2015.
 */

public class DetectMarks extends AsyncTask<Mat, Void, Void> {

    private boolean busy = true;
    private Mat frame = null;
    private List<MatOfPoint> squares = null;
    private int state = -1;

    public DetectMarks() {

    }

    public List<MatOfPoint> GetSquares(){
        return this.squares;
    }

    public Void doInBackground(Mat... params) {
        mainDetect(params[0], params[1]);
        this.busy = false;
        return null;
    }

    public Mat GetFrame(){
        return this.frame;
    }

    public boolean isBusy() {return this.busy;}

    public void mainDetect(Mat rgba, Mat gray){
        Size sizeRgba = rgba.size();
        Mat rgbaInnerWindow;
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;

        this.squares = Find_Squares(rgba);
    }

    private List<MatOfPoint> SquaresFilter(List<MatOfPoint> squares) {

        if (squares != null) {
            List<MatOfPoint> filters = new ArrayList<>();
            boolean[] sq = new boolean[squares.size()];
            Arrays.fill(sq, true);
            for (int i = 0; i < squares.size(); i++) {
                if (sq[i]) {
                    filters.add(squares.get(i));
                    sq[i] = false;
                }
                for (int j = 0; j < squares.size(); j++) {
                    if (sq[j]) {
                        double[] centerI = Center(squares.get(i)).get(0, 0);
                        double[] centerJ = Center(squares.get(j)).get(0, 0);
                        if (Math.abs(centerI[0] - centerJ[0]) < 20 &&
                                Math.abs(centerI[1] - centerJ[1]) < 20) {
                            sq[j] = false;
                            break;
                        }
                    }
                }
            }

            return filters;
        }

        return squares;
    }

    private MatOfPoint Center(MatOfPoint square){
        Point mid = new Point();
        mid.x = (square.get(0,0)[0] + square.get(2,0)[0]) / 2;
        mid.y = (square.get(0,0)[1] + square.get(2,0)[1]) / 2;
        return new MatOfPoint(mid);
    }

    private List<MatOfPoint> Find_Squares(Mat img) {
        Mat bin = new Mat();
        Mat cannyDst = new Mat();

        Imgproc.Canny(img, cannyDst, 0, 50);
        Mat dilated_ker = Mat.ones(1, 1, CvType.CV_32F);
        Imgproc.dilate(cannyDst, bin, dilated_ker);
        List<MatOfPoint> contours = new ArrayList<>();
        List<MatOfPoint> quads = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint cnt : contours) {
            MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
            double cnt_len = Imgproc.arcLength(cnt2f, true);
            Imgproc.approxPolyDP(cnt2f, cnt2f, 0.02 * cnt_len, true);
            cnt2f.convertTo(cnt, CvType.CV_32S);
            if (Imgproc.contourArea(cnt2f) > 10000){

                double[] cos = new double[4];

                for (int i = 0; i < 4; i++){
                    double[] p0 = cnt.get(i, 0);
                    double[] p1 = cnt.get((i + 1)%4, 0);
                    double[] p2 = cnt.get((i + 2)%4, 0);
                    cos[i] = angle_cos(p0, p1, p2);
                }

                if (getMax(cos) < 0.1){
                    quads.add(cnt);
                    return (quads);
                }

            }
        }

        return null;
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
}

