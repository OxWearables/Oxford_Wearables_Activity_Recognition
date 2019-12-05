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

    public static final int nFFT3D = 15;
    public static final int lowPassCutFrequency = 20;

    public static double[] extract(
        double[] xArray, double[] yArray, double[] zArray, int sampleRate)
    {
        double[] vArray = norm(xArray, yArray, zArray);

        double accPA = calculateAccPA(vArray, sampleRate);
        double[] basicStatistics = calculateBasicStatistics(
            xArray, yArray, zArray, sampleRate);
        double[] sanDiegoFeatures = calculateSanDiegoFeatures(
            xArray, yArray, zArray, sampleRate);
        double[] sanDiegoFFT = calculateSanDiegoFFT(vArray, sampleRate);
        double[] madFeatures = calculateMADFeatures(vArray);
        double[] unileverFeatures = calculateUnileverFeatures(vArray, sampleRate);
        double[] fft3D = calculateFFT3D(xArray, yArray, zArray, vArray, nFFT3D);

        double[] feats = new double[1 +
            basicStatistics.length +
            sanDiegoFeatures.length +
            sanDiegoFFT.length +
            madFeatures.length +
            unileverFeatures.length +
            fft3D.length];

        feats[0] = accPA;
        for (int i=0; i<basicStatistics.length; i++){
            feats[1 + i] = basicStatistics[i];
        }
        for (int i=0; i<sanDiegoFeatures.length; i++){
            feats[1 + basicStatistics.length + i] = sanDiegoFeatures[i];
        }
        for (int i=0; i<sanDiegoFFT.length; i++){
            feats[1 +
                basicStatistics.length +
                sanDiegoFeatures.length +
                i] = sanDiegoFFT[i];
        }
        for (int i=0; i<madFeatures.length; i++){
            feats[1 +
                basicStatistics.length +
                sanDiegoFeatures.length +
                sanDiegoFFT.length +
                i] = madFeatures[i];
        }
        for (int i=0; i<unileverFeatures.length; i++){
            feats[1 +
                basicStatistics.length +
                sanDiegoFeatures.length +
                sanDiegoFFT.length +
                madFeatures.length +
                i] = unileverFeatures[i];
        }
        for (int i=0; i<fft3D.length; i++){
            feats[
                1 +
                basicStatistics.length +
                sanDiegoFeatures.length +
                sanDiegoFFT.length +
                unileverFeatures.length +
                madFeatures.length +
                i] = fft3D[i];
        }

        return feats;

    }

    public static float[] extract(
        float[] xArray, float[] yArray, float[] zArray, int sampleRate
    )
    {
        double[] xArrayDouble = new double[xArray.length];
        double[] yArrayDouble = new double[yArray.length];
        double[] zArrayDouble = new double[zArray.length];
        for (int i=0; i<xArray.length; i++){
            xArrayDouble[i] = (double) xArray[i];
        }
        for (int i=0; i<yArray.length; i++){
            yArrayDouble[i] = (double) yArray[i];
        }
        for (int i=0; i<zArray.length; i++){
            zArrayDouble[i] = (double) zArray[i];
        }

        double[] featsDouble = extract(
            xArrayDouble, yArrayDouble, zArrayDouble, sampleRate);
        float[] feats = new float[featsDouble.length];
        for (int i=0; i<featsDouble.length; i++){
            feats[i] = (float) featsDouble[i];
        }
        return feats;
    }

    public static double calculateAccPA(double[] vArray, int sampleRate)
    {
        // Grab a copy because the next operations are inplace
        double[] vArrayNew = vArray.clone();

        // Remove gravity
        minusOne(vArrayNew);

        // Low-pass filter
        LowpassFilter filter = new LowpassFilter(lowPassCutFrequency, sampleRate);
        filter.filter(vArrayNew);

        // Truncate
        trunc(vArrayNew);

        // Mean
        double accPA = mean(vArrayNew);

        return accPA;

    }

    public static double [] calculateBasicStatistics(
        double[] xArray, double[] yArray, double[] zArray, int sampleRate
    )
    {
		// calculate raw x/y/z summary values
		double xMean = mean(xArray);
		double yMean = mean(yArray);
		double zMean = mean(zArray);
		double xRange = range(xArray);
		double yRange = range(yArray);
		double zRange = range(zArray);
		double xStd = std(xArray, xMean);
		double yStd = std(yArray, yMean);
		double zStd = std(zArray, zMean);

        double xyCovariance = covariance(xArray, yArray, xMean, yMean, 0);
        double xzCovariance = covariance(xArray, zArray, xMean, zMean, 0);
        double yzCovariance = covariance(yArray, zArray, yMean, zMean, 0);

        double[] feats = {
            xMean, yMean, zMean,
            xRange, yRange, zRange,
            xStd, yStd, zStd,
            xyCovariance, xzCovariance, yzCovariance,
        };

        return feats;

    }

    /* From paper:
     * Hip and Wrist Accelerometer Algorithms for Free-Living Behavior Classification.
     * Ellis K, Kerr J, Godbole S, Staudenmayer J, Lanckriet G.
     * https://www.ncbi.nlm.nih.gov/pubmed/26673126
     */
	public static double [] calculateSanDiegoFeatures(
        double[] xArray, double[] yArray, double[] zArray, int sampleRate)
        {

		int n = xArray.length;

		// San Diego g values
		// the g matric contains the estimated gravity vector, which is
		// essentially a low pass filter
		double[] gg = sanDiegoGetAvgGravity(xArray, yArray, zArray, sampleRate);
		double gxMean = gg[0];
		double gyMean = gg[1];
		double gzMean = gg[2];

		// subtract column means and get vector magnitude
		double[] v = new double[n]; // vector magnitude
		double[] wx = new double[n]; // gravity adjusted weights
		double[] wy = new double[n];
		double[] wz = new double[n];
		for (int i = 0; i < n; i++) {
			wx[i] = xArray[i]-gxMean;
			wy[i] = yArray[i]-gyMean;
			wz[i] = zArray[i]-gzMean;
			v[i] = norm( wx[i], wy[i], wz[i]);
		}

		// Write epoch
		double sdMean = mean(v);
		double sdStd = stdR(v, sdMean);
		double sdCoefVariation = 0.0;
		if (sdMean!=0) sdCoefVariation = sdStd/sdMean;
		// median, min, max, 25thp, 75thp
        double[] paQuartiles = percentiles(v, new double[] {0, 0.25, 0.5, 0.75, 1});

		double autoCorrelation = Correlation(v, v, sampleRate);
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

        double[] feats = {
            sdMean, sdStd, sdCoefVariation,
            paQuartiles[2], paQuartiles[0],
            paQuartiles[4], paQuartiles[1],
            paQuartiles[3],
            autoCorrelation,
            xyCorrelation, xzCorrelation, yzCorrelation,
            angleAvgStdYZ[0], angleAvgStdZX[0], angleAvgStdYX[0],
            angleAvgStdYZ[1], angleAvgStdZX[1], angleAvgStdYX[1],
            gxyAngle, gzxAngle, gyxAngle,
        };

        return feats;

	}

	// returns { x, y, z } averages of gravity
	public static double[] sanDiegoGetAvgGravity(
        double[] xArray, double[] yArray, double[] zArray, int sampleRate)
        {
		// San Diego paper 0.5Hz? low-pass filter approximation
		// this code takes in w and returns gg
		int n = xArray.length;
		int gn = n-(sampleRate-1); // number of moving average values to estimate gravity direction with
		int gStartIdx = n - gn; // number of gravity values to discard at beginning

		double[] gx = new double[gn];
		double[] gy = new double[gn];
		double[] gz = new double[gn];

        // calculating moving average of signal
        double x = 0.9;

        double xMovAvg = (1-x)*xArray[0];
        double yMovAvg = (1-x)*yArray[0];
        double zMovAvg = (1-x)*zArray[0];

        for (int c = 1; c < n; c++) {
            if (c < gStartIdx) {
                xMovAvg = xMovAvg * x + (1-x) * xArray[c];
                yMovAvg = yMovAvg * x + (1-x) * yArray[c];
                zMovAvg = zMovAvg * x + (1-x) * zArray[c];
            } else {
                // only store the signal after it has stabilized
                xMovAvg = xMovAvg * x + (1-x) * xArray[c];
                yMovAvg = yMovAvg * x + (1-x) * yArray[c];
                zMovAvg = zMovAvg * x + (1-x) * zArray[c];
                gx[c-gStartIdx] = xMovAvg;
                gy[c-gStartIdx] = yMovAvg;
                gz[c-gStartIdx] = zMovAvg;
            }
        }

		// column means
		double gxMean = mean(gx);
		double gyMean = mean(gy);
        double gzMean = mean(gz);

		return new double[] {gxMean, gyMean, gzMean};
	}

	public static double[] calculateSanDiegoFFT(double[] vArray, int sampleRate) {

		int n = vArray.length;
		// FFT frequency interval = sample frequency / num samples
		double FFTinterval = sampleRate / (1.0 * n); // (Hz)

		int numBins = 15; // From original implementation

		// set input data array
		double[] vmFFT = new double[n];
        for (int c = 0; c < n; c++) {
        	vmFFT[c] = vArray[c];
        }

        // Hanning window attenuates the signal to zero at it's start and end
        HanningWindow(vmFFT,n);

        DoubleFFT_1D transformer = new DoubleFFT_1D(n);
		transformer.realForward(vmFFT);
		double max = max(vmFFT);

		// find dominant frequency, second dominant frequency, and dominant between .6 - 2.5Hz
		double f1=-1, f33=-1;
		double p1=0, p33=0;

		double totalPLnP = 0.0; // sum of P * ln(P)
		double magDC = vmFFT[0]/max;
		totalPLnP += magDC * Math.log(magDC);

		for (int i = 1; i < n/2; i++) {
			double freq = FFTinterval * i;
			double Re = vmFFT[i*2];
			double Im = vmFFT[i*2+1];
			double mag = Math.sqrt( Re * Re + Im * Im)/max;

        	totalPLnP += mag * Math.log(mag);

        	if (mag>p1) {
        		f1 = freq;
        		p1 = mag;
        	}
        	if (freq > 0.3 && freq < 3 && mag>p33) {
        		f33 = freq;
        		p33 = mag;
        	}

		}
		// entropy, AKA (Power) Spectral Entropy, measures 'peakyness' of the frequency spectrum
		// This should be higher where there are periodic motions such as walking
		double H = - totalPLnP;// / total - (n/2) * Math.log(total);

		double[] binnedFFT = new double[numBins];

		for (int i = 0; i < numBins; i++) {
			binnedFFT[i] = 0;
		}
		int numWindows = (int) Math.floor(n/sampleRate);
		double[] windowSamples = new double[sampleRate];
        DoubleFFT_1D windowTransformer = new DoubleFFT_1D(sampleRate);
        max = Double.NEGATIVE_INFINITY;
        // do a FFT on each 1 second window (therefore FFT-interval will be one)
        FFTinterval = 1;
		for (int window = 0; window < numWindows; window++ ) {
			for (int i = 0; i < sampleRate; i++) {
				windowSamples[i] = vArray[i+window*(sampleRate/2)];
			}
			HanningWindow(windowSamples, sampleRate);
			windowTransformer.realForward(windowSamples);
			for (int i = 0; i < numBins; i++) {
				double mag;
				if (i==0) {
					mag = windowSamples[i];
				}
				else {
					double Re = windowSamples[i*2];
					double Im = windowSamples[i*2+1];
					mag = Math.sqrt( Re * Re + Im * Im);
				}
				binnedFFT[i] += mag;
				max = Math.max(max, mag); // find max as we go
			}
		}

		// Divide by the number of windows (to get the mean value)
		// Then divide by the maximum of the windowed FFT values (found before combination)
		// Note this does not mean the new max of binnedFFT will be one, it will be less than one if one window is stronger than the others
		scale(binnedFFT, 1 / (max * numWindows));

        // Concatenate results
        double[] feats = new double[5 + binnedFFT.length];
        feats[0] = f1; feats[1] = p1;
        feats[2] = f33; feats[3] = p33;
        feats[4] = H;
        for (int i=0; i < binnedFFT.length; i++) {
            feats[5+i] = binnedFFT[i];
        }

        return feats;

	}

	/* From paper:
	 * A universal, accurate intensity-based classification of different physical
	 * activities using raw data of accelerometer.
	 * Henri Vaha-Ypya, Tommi Vasankari, Pauliina Husu, Jaana Suni and Harri Sievanen
     * https://www.ncbi.nlm.nih.gov/pubmed/24393233
	 */
	public static double[] calculateMADFeatures(double[] vArray)
    {
		// used in calculation
		int n = vArray.length;
		double N = (double) n; // avoid integer arithmetic
		double vmMean = mean(vArray);
		double vmStd = std(vArray, vmMean);

		// features from paper:
		double MAD = 0; // Mean amplitude deviation (MAD) describes the typical distance of data points about the mean
		double MPD = 0; // Mean power deviation (MPD) describes the dispersion of data points about the mean
		double skew = 0; // Skewness (skewR) describes the asymmetry of dispersion of data points about the mean
		double kurt = 0; // Kurtosis (kurtR) describes the peakedness of the distribution of data points
		for (int c = 0; c < n; c++) {
			double diff = vArray[c] - vmMean;
			MAD += Math.abs(diff);
			MPD += Math.pow(Math.abs(diff), 1.5);
			skew += Math.pow(diff/vmStd, 3);
			kurt += Math.pow(diff/vmStd, 4);
		}

		MAD /= N;
		MPD /= Math.pow(N, 1.5);
		skew *= N / ((N-1)*(N-2));
        kurt = kurt * N*(N+1)/((N-1)*(N-2)*(N-3)*(N-4)) - 3*(N-1)*(N-1)/((N-2)*(N-3));

        double[] feats = { MAD, MPD, skew, kurt };

        return feats;

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
		return getFFTmagnitude(FFT, FFT.length);
	}

	/* converts FFT library's complex output to only absolute magnitude */
	public static double[] getFFTmagnitude(double[] FFT, int n) {
		if (n<1) {
			System.err.println("cannot get FFT magnitude of array with zero elements");
			return new double[] {0.0};
		}

		/*
		if n is even then
		 a[2*k] = Re[k], 0<=k<n/2
		 a[2*k+1] = Im[k], 0<k<n/2
		 a[1] = Re[n/2]
		e.g for n = 6: (yes there will be 7 array elements for 4 magnitudes)
		a = { Re[0], Re[3], Re[1], Im[1], Re[2], Im[2], Im[3]}

		if n is odd then
		 a[2*k] = Re[k], 0<=k<(n+1)/2
		 a[2*k+1] = Im[k], 0<k<(n-1)/2
		 a[1] = Im[(n-1)/2]
		e.g for n = 7: (again there will be 7 array elements for 4 magnitudes)
		a = { Re[0], Im[3], Re[1], Im[1], Re[2], Im[2], Re[3]}

		*/
		int m = 1 + (int) Math.floor(n/2); // output array size
		double[] output = new double[m];
		double Re, Im;


		output[0] = FFT[0];
		for (int i = 1; i < m-1; i++) {
			Re = FFT[i*2];
			Im = FFT[i*2 + 1];
			output[i] = Math.sqrt(Re * Re + Im * Im);
		}
		// highest frequency will be
		output[m-1] = Math.sqrt(FFT[1] * FFT[1] + FFT[m] * FFT[m]);
		return output;
	}

	/*
	 * Get FFT bins for each of the 3 axes, also combines them into 'mfft'.
	 */
	public static double[] calculateFFT3D(
        double[] xArray, double[] yArray, double[] zArray, double[] vArray, int nFFT)
    {
        double[] features = new double[4*nFFT];
		int n = xArray.length;
		DoubleFFT_1D transformer = new DoubleFFT_1D(n);
		double[] input = new double[n*2];

		int m = 1 + (int) Math.floor(n/2); // output array size
		double[] output = new double[m];

        for (int i=0; i<n; i++) {
        	input[i] = xArray[i];
        }
		HanningWindow(input, n);
		transformer.realForward(input);
        output = getFFTmagnitude(input, n);  //! use input instead of output?
        for (int i=0; i<nFFT; i++) {
            features[i] = output[i];
        }

        input = new double[n*2];
        for (int i=0; i<n; i++) {
        	input[i] = yArray[i];
        }
		HanningWindow(input, n);
		transformer.realForward(input);
        input = getFFTmagnitude(input, n);
        for (int i=0; i<nFFT; i++) {
            features[nFFT+i] = input[i];
        }

        input = new double[n*2];
        for (int i=0; i<n; i++) {
        	input[i] = zArray[i];
        }
		HanningWindow(input, n);
		transformer.realForward(input);
        input = getFFTmagnitude(input, n);
        for (int i=0; i<nFFT; i++) {
            features[2*nFFT+i] = input[i];
        }

        input = new double[n*2];
        for (int i=0; i<n; i++) {
            input[i] = vArray[i];
        }
        HanningWindow(input, n);
        transformer.realForward(input);
        input = getFFTmagnitude(input, n);
        for (int i=0; i<nFFT; i++) {
            features[3*nFFT+i] = input[i];
        }

        return features;

	}

    /* From paper:
	 * Physical Activity Classification using the GENEA Wrist Worn Accelerometer
	 * Shaoyan Zhang, Alex V. Rowlands, Peter Murray, Tina Hurst
     * https://www.ncbi.nlm.nih.gov/pubmed/21988935
	 */
    public static double [] calculateUnileverFeatures(double[] vArray, int sampleRate)
    {
		int n = vArray.length;
		double FFTinterval = sampleRate / (1.0 * n); // (Hz)
		double binSize = 0.1; // desired size of each FFT bin (Hz)
		double maxFreq = 15; // min and max for searching for dominant frequency
		double minFreq = 0.3;

		int numBins = (int) Math.ceil((maxFreq-minFreq)/binSize);

		DoubleFFT_1D transformer = new DoubleFFT_1D(n);
		double[] vmFFT = new double[n * 2];
		// set input data array
        for (int c = 0; c < n; c++) {
        	// NOTE: this code will generate peaks at 10Hz, and 2Hz (as expected).
        	// Math.sin(c * 2 * Math.PI * 10 / intendedSampleRate) + Math.cos(c * 2 * Math.PI * 2 / intendedSampleRate);
        	vmFFT[c] = vArray[c];
        }

		HanningWindow(vmFFT, n);
        // FFT
		transformer.realForward(vmFFT);

        // find dominant frequency, second dominant frequency, and dominant between .6 - 2.5Hz
 		double f1=-1, f2=-1, f625=-1, f33=-1;
 		double p1=0, p2=0, p625=0, p33=0;

 		double totalPower = 0;
		int out_n = (int) Math.ceil(n/2); // output array size

 		for (int i = 1; i < out_n; i++) {
 			double freq = FFTinterval * i;
 			if (freq<minFreq || freq>maxFreq) continue;
        	double mag = Math.sqrt(vmFFT[i*2]*vmFFT[i*2] + vmFFT[i*2+1]*vmFFT[i*2+1]);///(n/2);
        	totalPower += mag;
        	if (mag>p1) {
        		f2 = f1;
        		p2 = p1;
        		f1 = freq;
        		p1 = mag;
        	} else if (mag > p2) {
        		f2 = freq;
        		p2 = mag;
        	}
        	if (mag>p625 && freq > 0.6 && freq < 2.5) {
        		f625 = freq;
        		p625 = mag;
        	}

        	int w = 20;
            int a = (int) Math.round(Math.abs(mag)*10);
        }

        double[] feats = { f1, p1, f2, p2, f625, p625, totalPower };

        return feats;

	}

    private static double norm(double x, double y, double z)
    {
        if ((!Double.isNaN(x)) || (!Double.isNaN(y)) || (!Double.isNaN(z))) {
            return Math.sqrt(x * x + y * y + z * z);
        } else {
            return 0.0;
        }
    }

    public static double[] norm(double[] xArray, double[] yArray, double[] zArray)
    {
        double x, y, z;
        double[] arr = new double[xArray.length];
        for (int i=0; i<xArray.length; i++) {
            x = xArray[i];
            y = yArray[i];
            z = zArray[i];
            arr[i] = norm(x, y, z);
        }
        return arr;
    }

	private static void abs(double[] vals) {
		for (int c = 0; c < vals.length; c++) {
			vals[c] = Math.abs(vals[c]);
		}
	}

    private static void trunc(double[] vals) {
        for (int i=0; i<vals.length; i++) {
            vals[i] = Math.max(vals[i], 0);
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

	/*
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

    private static void minusOne(double[] vals){
        for (int i=0; i<vals.length; i++){
            vals[i] = vals[i] - 1;
        }
    }

}
