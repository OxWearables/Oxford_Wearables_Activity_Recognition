/* BSD 2-Clause (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
 *
 * Code for extracting features from raw triaxial acceleration measurements.
 * The features are derived from four papers:
 *
 * Hip and Wrist Accelerometer Algorithms for Free-Living Behavior
 * Classification. Ellis K, Kerr J, Godbole S, Staudenmayer J, Lanckriet G.
 *
 * A universal, accurate intensity-based classification of different
 * physical activities using raw data of accelerometer. Henri Vaha-Ypya,
 * Tommi Vasankari, Pauliina Husu, Jaana Suni and Harri Sievanen
 *
 * Physical Activity Classification using the GENEA Wrist Worn
 * Accelerometer Shaoyan Zhang, Alex V. Rowlands, Peter Murray, Tina Hurst
 *
 * Activity recognition using a single accelerometer placed at the wrist or ankle.
 * Mannini A, Intille SS, Rosenberger M, Sabatini AM, Haskell W.
 *
 * This code is distilled from
 * https://github.com/activityMonitoring/biobankAccelerometerAnalysis
 *
 */

import java.io.BufferedWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import org.jtransforms.fft.DoubleFFT_1D;

public class FeatureExtractor {


    public static double[] extract(final double[] x, final double[] y, final double[] z, final int sampleRate)
    {
        final double[] basicFeats = getBasic(x, y, z, sampleRate);
        final double[] sanDiegoFeats = getSanDiego(x, y, z, sampleRate);
        final double[] madFeats = getMAD(x, y, z);
        final double[] unileverFeats = getUnilever(x, y, z, sampleRate);
        final double[] FFT3D = getFFT3D(x, y, z);

        double[] feats = concat(basicFeats, sanDiegoFeats, madFeats, unileverFeats, FFT3D);

        return feats;
    }


    public static float[] extract(final float[] x, final float[] y, final float[] z, final int sampleRate)
    {
        double[] xDouble = new double[x.length];
        double[] yDouble = new double[y.length];
        double[] zDouble = new double[z.length];
        for (int i=0; i<x.length; i++) xDouble[i] = (double) x[i];
        for (int i=0; i<y.length; i++) yDouble[i] = (double) y[i];
        for (int i=0; i<z.length; i++) zDouble[i] = (double) z[i];

        double[] featsDouble = extract(xDouble, yDouble, zDouble, sampleRate);
        float[] feats = new float[featsDouble.length];
        for (int i=0; i<featsDouble.length; i++) feats[i] = (float) featsDouble[i];

        return feats;
    }


    /**
     * A bunch of basic statistics
     */
    public static double [] getBasic(double[] x, double[] y, double[] z, int sampleRate)
    {
        double enmoTrunc = getEnmoTrunc(x, y, z, sampleRate);
        // calculate raw x/y/z summary values
        double xMean = mean(x);
        double yMean = mean(y);
        double zMean = mean(z);
        double xRange = range(x);
        double yRange = range(y);
        double zRange = range(z);
        double xStd = std(x, xMean);
        double yStd = std(y, yMean);
        double zStd = std(z, zMean);

        double xyCovariance = covariance(x, y, xMean, yMean, 0);
        double xzCovariance = covariance(x, z, xMean, zMean, 0);
        double yzCovariance = covariance(y, z, yMean, zMean, 0);

        double[] feats = {
            enmoTrunc,
            xMean, yMean, zMean,
            xRange, yRange, zRange,
            xStd, yStd, zStd,
            xyCovariance, xzCovariance, yzCovariance,
        };

        return feats;
    }

    /**
     * Euclidean norm minus one truncated: max(0, sqrt(x^2+y^2+z^2)-1)
     */
    public static double getEnmoTrunc(double[] x, double[] y, double[] z, int sampleRate)
    {
        final int lowpassCutFrequency = 20;
        double[] v = norm(x, y, z);
        // Remove gravity
        for (int i=0; i<v.length; i++) v[i] -= 1;
        // Low-pass filter
        new LowpassFilter(lowpassCutFrequency, sampleRate).filter(v);
        // Truncate
        for (int i=0; i<v.length; i++) v[i] = Math.max(v[i], 0);
        // Mean
        double enmoTrunc = mean(v);
        return enmoTrunc;
    }


    /**
     * From paper: Hip and Wrist Accelerometer Algorithms for Free-Living Behavior
     * Classification. Ellis K, Kerr J, Godbole S, Staudenmayer J, Lanckriet G.
     * https://www.ncbi.nlm.nih.gov/pubmed/26673126
     */
    public static double [] getSanDiego(double[] x, double[] y, double[] z, int sampleRate)
    {
        final int n = x.length;

        double[] gg = getSanDiegoGravity(x, y, z, sampleRate);
        double gxMean = gg[0];
        double gyMean = gg[1];
        double gzMean = gg[2];

        // subtract column means and get vector magnitude
        double[] wv = new double[n]; // vector magnitude
        double[] wx = new double[n]; // gravity adjusted weights
        double[] wy = new double[n];
        double[] wz = new double[n];
        for (int i = 0; i < n; i++) {
            wx[i] = x[i]-gxMean;
            wy[i] = y[i]-gyMean;
            wz[i] = z[i]-gzMean;
            wv[i] = norm( wx[i], wy[i], wz[i]);
        }

        double sdMean = mean(wv);
        double sdStd = stdR(wv, sdMean);
        double sdCoefVariation = 0.0;
        if (sdMean!=0) sdCoefVariation = sdStd/sdMean;
        // median, min, max, 25thp, 75thp
        double[] paQuartiles = percentiles(wv, new double[] {0, 0.25, 0.5, 0.75, 1});

        double autoCorrelation = Correlation(wv, wv, sampleRate);
        double xyCorrelation = Correlation(wx, wy);
        double xzCorrelation = Correlation(wx, wz);
        double yzCorrelation = Correlation(wy, wz);

        // Roll, Pitch, Yaw
        double [] angleAvgStdYZ = angleAvgStd(wy, wz);
        double [] angleAvgStdZX = angleAvgStd(wz, wx);
        double [] angleAvgStdYX = angleAvgStd(wy, wx);

        // gravity component angles
        double gxyAngle = Math.atan2(gyMean,gzMean);
        double gzxAngle = Math.atan2(gzMean,gxMean);
        double gyxAngle = Math.atan2(gyMean,gxMean);

        double[] FFTfeats = getSanDiegoFFT(x, y, z, sampleRate);

        double[] feats = concat(
            new double[] {sdMean, sdStd, sdCoefVariation},
            new double[] {paQuartiles[2], paQuartiles[0], paQuartiles[4], paQuartiles[1], paQuartiles[3]},
            new double[] {autoCorrelation, xyCorrelation, xzCorrelation, yzCorrelation},
            new double[] {angleAvgStdYZ[0], angleAvgStdZX[0], angleAvgStdYX[0]},
            new double[] {angleAvgStdYZ[1], angleAvgStdZX[1], angleAvgStdYX[1]},
            new double[] {gxyAngle, gzxAngle, gyxAngle},
            FFTfeats
        );

        return feats;
    }


    public static double[] getSanDiegoGravity(double[] x, double[] y, double[] z, int sampleRate)
    {
        // San Diego paper 0.5Hz? low-pass filter approximation
        final int n = x.length;
        final int gn = n-(sampleRate-1); // number of moving average values to estimate gravity direction with
        final int gStartIdx = n - gn; // number of gravity values to discard at beginning

        double[] gx = new double[gn];
        double[] gy = new double[gn];
        double[] gz = new double[gn];

        // calculating moving average of signal
        final double alpha = 0.9;

        double xMovAvg = (1-alpha)*x[0];
        double yMovAvg = (1-alpha)*y[0];
        double zMovAvg = (1-alpha)*z[0];

        for (int i = 1; i < n; i++) {
            if (i < gStartIdx) {
                xMovAvg = xMovAvg * alpha + (1-alpha) * x[i];
                yMovAvg = yMovAvg * alpha + (1-alpha) * y[i];
                zMovAvg = zMovAvg * alpha + (1-alpha) * z[i];
            } else {
                // only store the signal after it has stabilized
                xMovAvg = xMovAvg * alpha + (1-alpha) * x[i];
                yMovAvg = yMovAvg * alpha + (1-alpha) * y[i];
                zMovAvg = zMovAvg * alpha + (1-alpha) * z[i];
                gx[i-gStartIdx] = xMovAvg;
                gy[i-gStartIdx] = yMovAvg;
                gz[i-gStartIdx] = zMovAvg;
            }
        }

        // column means
        double gxMean = mean(gx);
        double gyMean = mean(gy);
        double gzMean = mean(gz);

        return new double[] {gxMean, gyMean, gzMean};
    }


    public static double[] getSanDiegoFFT(double[] x, double[] y, double[] z, int sampleRate)
    {

        /*
        Compute FFT and power spectrum density
        */
        final double[] v = norm(x, y, z);
        final int n = v.length;
        final double vMean = mean(v);
        // Initialize FFT array. Also zero-center signal to remove 0Hz frequency.
        //TODO: Zero-padding for more accurate frequency estimation:  https://uk.mathworks.com/help/signal/ug/amplitude-estimation-and-zero-padding.html
        // Note: Padding before or after windowing?: https://dsp.stackexchange.com/questions/13736/zero-pad-before-or-after-windowing-for-fft
        double[] vFFT = new double[n];
        for (int i = 0; i < n; i++)  vFFT[i] = v[i] - vMean;
        // Hanning window attenuates the signal to zero at endpoints
        HanningWindow(vFFT, vFFT.length);
        new DoubleFFT_1D(vFFT.length).realForward(vFFT);
        final double[] vFFTpow = getFFTpower(vFFT);

        /*
        Compute spectral entropy
        See https://www.mathworks.com/help/signal/ref/pentropy.html#mw_a57f549d-996c-47d9-8d45-e80cb739ed41
        Note: the following is the spectral entropy of only half of the spectrum (which is symetric anyways)
        */
        double H = 0.0;  // spectral entropy
        final double vFFTpowsum = sum(vFFTpow);
        for (int i = 0; i < vFFTpow.length; i++) {
            double p = vFFTpow[i] / (vFFTpowsum + 1E-8);
            if (p <= 0) continue;
            H += -p * Math.log(p + 1E-8);
        }
        H /= Math.log(vFFTpow.length);  // Normalize spectral entropy

        /*
        Find dominant frequencies overall, also between 0.3Hz and 3Hz
        */
        final double FFTinterval = sampleRate / (1.0 * n); // (Hz)
        double f1=-1, f33=-1;
        double p1=0, p33=0;
        for (int i = 0; i < vFFTpow.length; i++) {
            double freq = FFTinterval * i;
            double p = vFFTpow[i];
            if (p > p1) {
                f1 = freq;
                p1 = p;
            }
            if (freq > 0.3 && freq < 3 && p > p33) {
                f33 = freq;
                p33 = p;
            }
        }
        // Use logscale for convenience
        p1 = Math.log(p1 + 1E-8);
        p33 = Math.log(p33 + 1E-8);

        /*
        Estimate powers for bins 0-14Hz using Welch's method
        See: https://en.wikipedia.org/wiki/Welch%27s_method
        Note: Averaging the magnitudes (instead of the powers) yielded
        slightly better classification results in random forest
        */
        final int numBins = 15;
        double[] binnedFFT = new double[numBins];
        for (int i = 0; i < numBins; i++) binnedFFT[i] = 0;
        final int windowOverlap = sampleRate / 2;  // 50% overlapping windows
        final int numWindows = n / windowOverlap - 1;
        double[] windowFFT = new double[sampleRate];
        DoubleFFT_1D windowTransformer = new DoubleFFT_1D(sampleRate);
        for (int i = 0; i < numWindows; i++ ) {
            for (int j = 0; j < windowFFT.length; j++) windowFFT[j] = v[i*windowOverlap+j];  // grab window
            HanningWindow(windowFFT, windowFFT.length);
            windowTransformer.realForward(windowFFT);
            final double[] windowFFTmag = getFFTmagnitude(windowFFT);
            for (int j = 0; j < binnedFFT.length; j++) binnedFFT[j] += windowFFTmag[j];
        }
        // Average magnitudes. Also use logscale for convenience.
        for (int i = 0; i < binnedFFT.length; i++) binnedFFT[i] = Math.log(binnedFFT[i]/numWindows + 1E-8);

        double[] feats = concat(new double[] {f1, p1, f33, p33, H}, binnedFFT);

        return feats;

    }


    /**
     * From paper:
     * A universal, accurate intensity-based classification of different physical
     * activities using raw data of accelerometer.
     * Henri Vaha-Ypya, Tommi Vasankari, Pauliina Husu, Jaana Suni and Harri Sievanen
     * https://www.ncbi.nlm.nih.gov/pubmed/24393233
     */
    public static double[] getMAD(double[] x, double[] y, double[] z)
    {
        final double[] v = norm(x, y, z);
        final int n = v.length;
        final double N = (double) n; // avoid integer arithmetic

        double vMean = mean(v);
        double vStd = std(v, vMean);
        double MAD = 0; // Mean amplitude deviation (MAD) describes the typical distance of data points about the mean
        double MPD = 0; // Mean power deviation (MPD) describes the dispersion of data points about the mean
        double skew = 0; // Skewness (skewR) describes the asymmetry of dispersion of data points about the mean
        double kurt = 0; // Kurtosis (kurtR) describes the peakedness of the distribution of data points
        for (int i = 0; i < n; i++) {
            double diff = v[i] - vMean;
            MAD += Math.abs(diff);
            MPD += Math.pow(Math.abs(diff), 1.5);
            skew += Math.pow(diff/(vStd + 1E-8), 3);
            kurt += Math.pow(diff/(vStd + 1E-8), 4);
        }

        MAD /= N;
        MPD /= Math.pow(N, 1.5);
        skew *= N / ((N-1)*(N-2));
        kurt = kurt * N*(N+1)/((N-1)*(N-2)*(N-3)*(N-4)) - 3*(N-1)*(N-1)/((N-2)*(N-3));

        double[] feats = { MAD, MPD, skew, kurt };

        return feats;
    }


    /**
     * From paper:
     * Physical Activity Classification using the GENEA Wrist Worn Accelerometer
     * Shaoyan Zhang, Alex V. Rowlands, Peter Murray, Tina Hurst
     * https://www.ncbi.nlm.nih.gov/pubmed/21988935
     * See also:
     * Activity recognition using a single accelerometer placed at the wrist or ankle.
     * Mannini A, Intille SS, Rosenberger M, Sabatini AM, Haskell W.
     */
    public static double [] getUnilever(double[] x, double[] y, double[] z, int sampleRate)
    {

        /*
        Compute FFT and power spectrum density
        */
        final double[] v = norm(x, y, z);
        final int n = v.length;
        final double vMean = mean(v);
        // Initialize FFT array. Also zero-center signal to remove 0Hz frequency.
        //TODO: Zero-padding for more accurate frequency estimation:  https://uk.mathworks.com/help/signal/ug/amplitude-estimation-and-zero-padding.html
        // Note: Padding before or after windowing?: https://dsp.stackexchange.com/questions/13736/zero-pad-before-or-after-windowing-for-fft
        double[] vFFT = new double[n];
        for (int i = 0; i < n; i++)  vFFT[i] = v[i] - vMean;
        // Hanning window attenuates the signal to zero at endpoints
        HanningWindow(vFFT, vFFT.length);
        new DoubleFFT_1D(vFFT.length).realForward(vFFT);
        final double[] vFFTpow = getFFTpower(vFFT);

        /*
        Find dominant frequencies between 0.3Hz - 15Hz, also between 0.6Hz - 2.5Hz
        */
        final double maxFreq = 15;
        final double minFreq = 0.3;
        final double FFTinterval = sampleRate / (1.0 * n); // (Hz)
         double f1=-1, f2=-1, f625=-1, f33=-1;
         double p1=0, p2=0, p625=0, p33=0;
        double totalPower = 0.0;
        for (int i = 0; i < vFFTpow.length; i++) {
            double freq = FFTinterval * i;
            if (freq < minFreq || freq > maxFreq) continue;
            double p = vFFTpow[i];
            totalPower += p;
            if (p > p1) {
                f2 = f1;
                p2 = p1;
                f1 = freq;
                p1 = p;
            } else if (p > p2) {
                f2 = freq;
                p2 = p;
            }
            if (p > p625 && freq > 0.6 && freq < 2.5) {
                f625 = freq;
                p625 = p;
            }
        }
        // Use logscale for convenience
        totalPower = Math.log(totalPower + 1E-8);
        p1 = Math.log(p1 + 1E-8);
        p2 = Math.log(p2 + 1E-8);
        p625 = Math.log(p625 + 1E-8);

        double[] feats = { f1, p1, f2, p2, f625, p625, totalPower };

        return feats;
    }

    /**
     * FFT over each axis and vector norm
     */
    public static double[] getFFT3D(double[] x, double[] y, double[] z)
    {
        final int nFFT = 15;
        final double[] v = norm(x, y, z);
        final int n = v.length;
        final DoubleFFT_1D transformer = new DoubleFFT_1D(2*n);

        double[] FFT = new double[2*n];  // zero-padded
        double[] features = new double[4*nFFT];
        double[] FFTmag = new double[n];
        
        // FFT along x-axis
        for (int i=0; i<FFT.length; i++) { FFT[i] = 0; }
        for (int i=0; i<n; i++) { FFT[i] = x[i]; }
        HanningWindow(FFT, n);
        transformer.realForward(FFT);
        FFTmag = getFFTmagnitude(FFT);
        for (int i=0; i<nFFT; i++) { features[i] = FFTmag[i]; }

        // FFT along y-axis
        for (int i=0; i<FFT.length; i++) { FFT[i] = 0; }
        for (int i=0; i<n; i++) { FFT[i] = y[i]; }
        HanningWindow(FFT, n);
        transformer.realForward(FFT);
        FFTmag = getFFTmagnitude(FFT);
        for (int i=0; i<nFFT; i++) { features[nFFT+i] = FFTmag[i]; }

        // FFT along z-axis
        for (int i=0; i<FFT.length; i++) { FFT[i] = 0; }
        for (int i=0; i<n; i++) { FFT[i] = z[i]; }
        HanningWindow(FFT, n);
        transformer.realForward(FFT);
        FFTmag = getFFTmagnitude(FFT);
        for (int i=0; i<nFFT; i++) { features[2*nFFT+i] = FFTmag[i]; }

        // FFT along vector norm
        for (int i=0; i<FFT.length; i++) { FFT[i] = 0; }
        for (int i=0; i<n; i++) { FFT[i] = v[i]; }
        HanningWindow(FFT, n);
        transformer.realForward(FFT);
        FFTmag = getFFTmagnitude(FFT);
        for (int i=0; i<nFFT; i++) { features[3*nFFT+i] = FFTmag[i]; }

        return features;

    }


    public static double[] HanningWindow(double[] signal_in, int size)
    {
        for (int i = 0; i < size; i++)
        {
            signal_in[i] = (double) (signal_in[i] * 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (size-1))));
        }
        return signal_in;
    }

    public static double[] getFFTmagnitude(double[] FFT) {
        return getFFTmagnitude(FFT, true);
    }

    /* converts FFT library's complex output to only absolute magnitude */
    public static double[] getFFTmagnitude(double[] FFT, boolean normalize) {
        /* Get magnitudes from FFT coefficients */

        double[] FFTmag = getFFTpower(FFT, normalize);
        for (int i=0; i<FFTmag.length; i++) FFTmag[i] = Math.sqrt(FFTmag[i]);
        return FFTmag;
    }

    public static double[] getFFTpower(double[] FFT) {
        return getFFTpower(FFT, true);
    }

    public static double[] getFFTpower(double[] FFT, boolean normalize) {
        /*
        Get powers from FFT coefficients

        The layout of FFT is as follows (computed using JTransforms, see
        https://github.com/wendykierp/JTransforms/blob/3c3253f240510c5f9ec700f2d9d25cfadfc857cc/src/main/java/org/jtransforms/fft/DoubleFFT_1D.java#L459):

        If n is even then
        FFT[2*k] = Re[k], 0<=k<n/2
        FFT[2*k+1] = Im[k], 0<k<n/2
        FFT[1] = Re[n/2]
        e.g. for n=6:
        FFT = { Re[0], Re[3], Re[1], Im[1], Re[2], Im[2] }

        If n is odd then
        FFT[2*k] = Re[k], 0<=k<(n+1)/2
        FFT[2*k+1] = Im[k], 0<k<(n-1)/2
        FFT[1] = Im[(n-1)/2]
        e.g for n = 7:
        FFT = { Re[0], Im[3], Re[1], Im[1], Re[2], Im[2], Re[3] }

        See also: https://stackoverflow.com/a/5010434/3250500
        */

        final int n = FFT.length;
        final int m = (int) Math.ceil((double) n / 2);
        double[] FFTpow = new double[m];
        double Re, Im;

        FFTpow[0] = FFT[0] * FFT[0];
        for (int i = 1; i < m-1; i++) {
            Re = FFT[i*2];
            Im = FFT[i*2 + 1];
            FFTpow[i] = Re * Re + Im * Im;
        }
        // The last power is a bit tricky due to the weird layout of FFT
        if (n % 2 == 0) {
            Re = FFT[n-2];  // FFT[2*m-2]
            Im = FFT[n-1];  // FFT[2*m-1]
        } else {
            Re = FFT[n-1];  // FFT[2*m-2]
            Im = FFT[1];
        }
        FFTpow[m-1] = Re * Re + Im * Im;

        if (normalize) {
            // Divide by length of the signal
            for (int i=0; i<m; i++) FFTpow[i] /= n;
        }

        return FFTpow;
    }

    private static double norm(double x, double y, double z)
    {
        if ((!Double.isNaN(x)) || (!Double.isNaN(y)) || (!Double.isNaN(z))) {
            return Math.sqrt(x * x + y * y + z * z);
        } else {
            return 0.0;
        }
    }

    public static double[] norm(double[] x, double[] y, double[] z)
    {
        double xi, yi, zi;
        double[] v = new double[x.length];
        for (int i=0; i<x.length; i++) {
            xi = x[i];
            yi = y[i];
            zi = z[i];
            v[i] = norm(xi, yi, zi);
        }
        return v;
    }

    private static void abs(double[] vals) {
        for (int c = 0; c < vals.length; c++) {
            vals[c] = Math.abs(vals[c]);
        }
    }

    private static double sum(double[] vals) {
        if (vals.length == 0) {
            return Double.NaN;
        }
        double sum = 0;
        for (int c = 0; c < vals.length; c++) {
            if (!Double.isNaN(vals[c])) {
                sum += vals[c];
            }
        }
        return sum;
    }

    private static double mean(double[] vals) {
        if (vals.length == 0) {
            return Double.NaN;
        }
        return sum(vals) / (double) vals.length;
    }

    private static double mean(List<Double> vals) {
        if (vals.size() == 0) {
            return Double.NaN;
        }
        return sum(vals) / (double) vals.size();
    }

    private static double sum(List<Double> vals) {
        if (vals.size() == 0) {
            return Double.NaN;
        }
        double sum = 0;
        for (int c = 0; c < vals.size(); c++) {
            sum += vals.get(c);
        }
        return sum;
    }

    private static double range(double[] vals) {
        if (vals.length == 0) {
            return Double.NaN;
        }
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (int c = 0; c < vals.length; c++) {
            if (vals[c] < min) {
                min = vals[c];
            }
            if (vals[c] > max) {
                max = vals[c];
            }
        }
        return max - min;
    }

    // standard deviation
    private static double std(double[] vals, double mean) {
        if (vals.length == 0) {
            return Double.NaN;
        }
        double var = 0; // variance
        double len = vals.length; // length
        for (int c = 0; c < len; c++) {
            if (!Double.isNaN(vals[c])) {
                var += Math.pow((vals[c] - mean), 2);
            }
        }
        return Math.sqrt(var / len);
    }

    // same as above but matches R's (n-1) denominator (Bessel's correction)
    private static double stdR(double[] vals, double mean) {
        if (vals.length == 0) {
            return Double.NaN;
        }
        double var = 0; // variance
        double len = vals.length; // length
        for (int c = 0; c < len; c++) {
            if (!Double.isNaN(vals[c])) {
                var += Math.pow((vals[c] - mean), 2);
            }
        }
        return Math.sqrt(var / (len-1));
    }

    private static double Correlation(double[] vals1, double[] vals2) {
        return Correlation(vals1, vals2, 0);
    }

    private static double Correlation(double[] vals1, double[] vals2, int lag) {
        lag = Math.abs(lag); // should be identical
        if ( vals1.length <= lag || vals1.length != vals2.length ) {
            return Double.NaN;
        }
        double sx = 0.0;
        double sy = 0.0;
        double sxx = 0.0;
        double syy = 0.0;
        double sxy = 0.0;
        final double small = 1e-16;

        int nmax = vals1.length;
        int n = nmax - lag;

        for(int i = lag; i < nmax; ++i) {
            double x = vals1[i-lag];
            double y = vals2[i];

            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }

        // covariation
        double cov = sxy / n - sx * sy / n / n;
        // standard error of x
        double sigmax = Math.sqrt(Math.max(sxx / n -  sx * sx / n / n, 0));
        // standard error of y
        double sigmay = Math.sqrt(Math.max(syy / n -  sy * sy / n / n, 0));

        // correlation is just a normalized covariation
        return cov / (sigmax * sigmay + small);
    }

    // covariance of two signals (with lag in samples)
    private static double covariance(double[] vals1, double[] vals2, double mean1, double mean2, int lag) {
        lag = Math.abs(lag); // should be identical
        if ( vals1.length <= lag || vals1.length != vals2.length ) {
            return Double.NaN;
        }
        double cov = 0; // covariance
        for (int c = lag; c < vals1.length; c++) {
            if (!Double.isNaN(vals1[c-lag]) && !Double.isNaN(vals2[c])) {
                cov += (vals1[c]-mean1) * (vals2[c]-mean2);
            }
        }
        cov /= vals1.length+1-lag;
        return cov;
    }

    /**
     * Implementation of the following features aims to match the paper:
     * Hip and Wrist Accelerometer Algorithms for Free-Living Behavior Classification
     * Katherine Ellis, Jacqueline Kerr, Suneeta Godbole, John Staudenmayer, and Gert Lanckriet
     */

    // percentiles = {0.25, 0.5, 0.75}, to calculate 25th, median and 75th percentile
    private static double[] percentiles(double[] vals, double[] percentiles) {
        double[] output = new double[percentiles.length];
        int n = vals.length;
        if (n == 0) {
            Arrays.fill(output, Double.NaN);
            return output;
        }
        if (n == 1) {
            Arrays.fill(output, vals[0]);
            return output;
        }
        double[] sortedVals = vals.clone();
        Arrays.sort(sortedVals);
        for (int i = 0; i<percentiles.length; i++) {
            // this follows the R default (R7) interpolation model
            // https://en.wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
            double h = percentiles[i] * (n-1) + 1;
            if (h<=1.0) {
                output[i] = sortedVals[0];
                continue;
            }
            if (h>=n) {
                output[i] = sortedVals[n-1];
                continue;
            }
            // interpolate using: x[h] + (h - floor(h)) (x[h + 1] - x[h])
            int hfloor = (int) Math.floor(h);
            double xh = sortedVals[hfloor-1] ;
            double xh2 = sortedVals[hfloor] ;
            output[i] = xh + (h - hfloor) * (xh2 - xh);
        }
        return output;
    }

    // returns {mean, standard deviation} together to reduce processing time
    private static double[] angleAvgStd(double[] vals1, double[] vals2) {
        int len = vals1.length;
        if ( len < 2 || len != vals2.length ) {
            return new double[] {Double.NaN, Double.NaN};
        }
        double[] angles = new double[len];
        double total = 0.0;
        for (int c = 0; c < len; c++) {
            angles[c] = Math.atan2(vals1[c],vals2[c]);
            total += angles[c];
        }
        double mean = total/len;
        double var = 0.0;
        for (int c = 0; c < len; c++) {
            var += Math.pow(angles[c] - mean, 2);
        }
        double std = Math.sqrt(var/(len-1)); // uses R's (n-1) denominator standard deviation (Bessel's correction)
        return new double[] {mean, std};
    }

    private static double correlation(
        double[] vals1, double[] vals2, double mean1, double mean2, int lag) {
        return covariance(vals1, vals2, mean1, mean2, lag)/(mean1*mean2);
    }

    private static double max(double[] vals) {
        double max = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < vals.length; c++) {
            max = Math.max(vals[c], max);
        }
        return max;
    }

    private static void scale(double[] vals, double scale) {
        for (int c = 0; c < vals.length; c++) {
            vals[c] = vals[c] * scale;
        }
    }

    private static double[] concat(double[]... arrays) {
        int length = 0;
        for (double[] array : arrays) length += array.length;
        double[] result = new double[length];
        int pos = 0;
        for (double[] array: arrays) {
            for (double element : array) {
                result[pos] = element;
                pos++;
            }
        }
        return result;
    }

}
