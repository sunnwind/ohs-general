package ohs.math;

import ohs.utils.Conditions;

/**
 * @author Heung-Seon Oh
 * 
 */
public class CommonMath {
	public static final double LOG_2_OF_E = 1f / Math.log(2);

	public static final double SQUARE_ROOT_OF_TWO_PI = Math.sqrt(2 * Math.PI);

	public static double beta(double a, double b) {
		return Math.exp(logBeta(a, b));
	}

	public static double binarySigmoid(double alpha, double x) {
		return 1 / (1 + Math.exp(-alpha * x));
	}

	public static double binarySigmoidDerivative(double alpha, double x) {
		return alpha * binarySigmoid(alpha, x) * (1 - binarySigmoid(alpha, x));
	}

	public static double bipolarSigmoid(double alpha, double x) {
		return (2 / (1 + Math.exp(-alpha * x))) - 1;
	}

	/**
	 * @param N11
	 *            number of documents that Category (+), Term (+)
	 * @param N10
	 *            number of documents that Category (+), Term (-)
	 * @param N01
	 *            number of documents that Category (-), Term (+)
	 * @param N00
	 *            number of documents that Category (-), Term (-)
	 * @return
	 */
	public static double chisquare(double N11, double N10, double N01, double N00) {
		double numerator = (N11 + N10 + N01 + N00) * Math.pow(N11 * N00 - N10 * N01, 2);
		double denominator = (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00);
		double chisquare = 0;

		if (numerator > 0 && denominator > 0) {
			chisquare = numerator / denominator;
		}
		return chisquare;
	}

	/**
	 * 
	 * Maths.java in mallet
	 * 
	 * @param n
	 * @param r
	 * @return
	 */
	public static double combination(int n, int r) {
		return Math.exp(logFactorial(n) - logFactorial(r) - logFactorial(n - r));
	}

	public static double divide(double numerator, double denominator) {
		double ret = 0;
		if (numerator > 0 && denominator > 0) {
			ret = numerator / denominator;
		}
		return ret;
	}

	public static double factorial(int n) {
		return Math.exp(logGamma(n + 1));
	}

	/**
	 * https://codingforspeed.com/using-faster-exponential-approximation/
	 * 
	 * @param x
	 * @return
	 */
	public static double fastExp(double x) {
		if (Double.isFinite(x)) {
			if (x == 0) {
				x = 1;
			} else {
				x = 1.0 + (x / 1024);
				int times = 10;
				double tmp = x;
				for (int i = 0; i < times; i++) {
					tmp *= tmp;
					if (!Double.isFinite(tmp)) {
						break;
					}
					x = tmp;
				}
			}
		}
		return x;
	}

	public static double fastExp2(double val) {
		final long tmp = (long) (1512775 * val + 1072632447);
		return Double.longBitsToDouble(tmp << 32);
	}

	public static double fastLog(double x) {
		return 6 * (x - 1) / (x + 1 + 4 * (Math.sqrt(x)));
	}

	public static double fastPow(double a, double b) {
		final long tmp = (long) (9076650 * (a - 1) / (a + 1 + 4 * (Math.sqrt(a))) * b + 1072632447);
		return Double.longBitsToDouble(tmp << 32);
	}

	public static double fastSigmoid(double x) {
		return 1 / (1 + fastExp(-x));
	}

	public static double gamma(double x) {
		return Math.exp(logGamma(x));
	}

	public static double gaussian(double x, double mu, double sigma) {
		double term1 = 1f / (SQUARE_ROOT_OF_TWO_PI * sigma);
		double term2 = Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
		return term1 * term2;
	}

	public static double hardSigmoid(double x) {
		return Math.max(0, Math.min(1, x * 0.2 + 0.5));
	}

	public static double hardTanh(double x) {
		double ret = x;
		if (x < -1) {
			ret = -1;
		} else if (x > 1) {
			ret = 1;
		}
		return ret;
	}

	public static double hardTanhGradient(double x) {
		double ret = 0;
		if (x >= -1 && x <= 1) {
			ret = 1;
		}
		return ret;
	}

	/**
	 * @param k
	 *            free parameter 30<= k <= 100
	 * @param b
	 *            free parameter b ~0.5
	 * @param T
	 *            the number of tokens in the collection
	 * @return
	 */
	public static double heapsLaw(double k, double b, double T) {
		return k * Math.pow(T, b);
	}

	public static boolean isEqual(double a, double b) {
		return a == b;
	}

	public static boolean isEqual(int a, int b) {
		return a == b;
	}

	public static double leakReLU(double x) {
		return leakReLU(x, 0.00000001);
	}

	public static double leakReLU(double x, double k) {
		return Math.max(x, x * k);
	}

	public static double leakReLUGradient(double x) {
		return leakReLUGradient(x, 0.00000001);
	}

	public static double leakReLUGradient(double fx, double k) {
		return fx > 0 ? 1 : k;
	}

	public static double log(double x) {
		return Math.log(x);
	}

	public static final double log2(double value) {
		return Math.log(value) * LOG_2_OF_E;
	}

	public static double logBeta(double a, double b) {
		return logGamma(a) + logGamma(b) - logGamma(a + b);
	}

	/**
	 * 
	 */
	//
	/**
	 * Maths.java in mallet
	 * 
	 * Copied as the "classic" method from Catherine Loader. Fast and Accurate
	 * Computation of Binomial Probabilities. 2001. (This is not the fast and
	 * accurate version.)
	 * 
	 * Computes p(x;n,p) where x~B(n,p)
	 * 
	 * @param x
	 * @param n
	 * @param p
	 * @return
	 */
	public static double logBinom(int x, int n, double p) {
		return logFactorial(n) - logFactorial(x) - logFactorial(n - x) + (x * Math.log(p))
				+ ((n - x) * Math.log(1 - p));
	}

	public static double logFactorial(int n) {
		return logGamma(n + 1);
	}

	/**
	 * From libbow, dirichlet.c Written by Tom Minka <minka@stat.cmu.edu>
	 * 
	 * Maths.java in mallet
	 * 
	 * @param x
	 * @return
	 */
	public static final double logGamma(double x) {
		double result, y, xnum, xden;
		int i;
		final double d1 = -5.772156649015328605195174e-1;
		final double p1[] = { 4.945235359296727046734888e0, 2.018112620856775083915565e2, 2.290838373831346393026739e3,
				1.131967205903380828685045e4, 2.855724635671635335736389e4, 3.848496228443793359990269e4,
				2.637748787624195437963534e4, 7.225813979700288197698961e3 };
		final double q1[] = { 6.748212550303777196073036e1, 1.113332393857199323513008e3, 7.738757056935398733233834e3,
				2.763987074403340708898585e4, 5.499310206226157329794414e4, 6.161122180066002127833352e4,
				3.635127591501940507276287e4, 8.785536302431013170870835e3 };
		final double d2 = 4.227843350984671393993777e-1;
		final double p2[] = { 4.974607845568932035012064e0, 5.424138599891070494101986e2, 1.550693864978364947665077e4,
				1.847932904445632425417223e5, 1.088204769468828767498470e6, 3.338152967987029735917223e6,
				5.106661678927352456275255e6, 3.074109054850539556250927e6 };
		final double q2[] = { 1.830328399370592604055942e2, 7.765049321445005871323047e3, 1.331903827966074194402448e5,
				1.136705821321969608938755e6, 5.267964117437946917577538e6, 1.346701454311101692290052e7,
				1.782736530353274213975932e7, 9.533095591844353613395747e6 };
		final double d4 = 1.791759469228055000094023e0;
		final double p4[] = { 1.474502166059939948905062e4, 2.426813369486704502836312e6, 1.214755574045093227939592e8,
				2.663432449630976949898078e9, 2.940378956634553899906876e10, 1.702665737765398868392998e11,
				4.926125793377430887588120e11, 5.606251856223951465078242e11 };
		final double q4[] = { 2.690530175870899333379843e3, 6.393885654300092398984238e5, 4.135599930241388052042842e7,
				1.120872109616147941376570e9, 1.488613728678813811542398e10, 1.016803586272438228077304e11,
				3.417476345507377132798597e11, 4.463158187419713286462081e11 };
		final double c[] = { -1.910444077728e-03, 8.4171387781295e-04, -5.952379913043012e-04, 7.93650793500350248e-04,
				-2.777777777777681622553e-03, 8.333333333333333331554247e-02, 5.7083835261e-03 };
		final double a = 0.6796875;

		if ((x <= 0.5) || ((x > a) && (x <= 1.5))) {
			if (x <= 0.5) {
				result = -Math.log(x);
				/* IntListMap whether X < machine epsilon. */
				if (x + 1 == 1) {
					return result;
				}
			} else {
				result = 0;
				x = (x - 0.5) - 0.5;
			}
			xnum = 0;
			xden = 1;
			for (i = 0; i < 8; i++) {
				xnum = xnum * x + p1[i];
				xden = xden * x + q1[i];
			}
			result += x * (d1 + x * (xnum / xden));
		} else if ((x <= a) || ((x > 1.5) && (x <= 4))) {
			if (x <= a) {
				result = -Math.log(x);
				x = (x - 0.5) - 0.5;
			} else {
				result = 0;
				x -= 2;
			}
			xnum = 0;
			xden = 1;
			for (i = 0; i < 8; i++) {
				xnum = xnum * x + p2[i];
				xden = xden * x + q2[i];
			}
			result += x * (d2 + x * (xnum / xden));
		} else if (x <= 12) {
			x -= 4;
			xnum = 0;
			xden = -1;
			for (i = 0; i < 8; i++) {
				xnum = xnum * x + p4[i];
				xden = xden * x + q4[i];
			}
			result = d4 + x * (xnum / xden);
		}
		/* X > 12 */
		else {
			y = Math.log(x);
			result = x * (y - 1) - y * 0.5 + .9189385332046727417803297;
			x = 1 / x;
			y = x * x;
			xnum = c[6];
			for (i = 0; i < 6; i++) {
				xnum = xnum * y + c[i];
			}
			xnum *= x;
			result += xnum;
		}
		return result;
	}

	public static double loglog(double x) {
		return Math.log(Math.log(x));
	}

	public static final double loglog2(double x) {
		return log2(log2(x));
	}

	// public static double sigmoid(double x) {
	// return binarySigmoid(1, x);
	// }

	public static void main(String[] args) {
		System.out.println("process begins.");

		{
			double cnt_x = 1938;
			double cnt_y = 1311;
			double cnt_xy = 1159;
			double total_cnt = 50000952;
			System.out.println(pmi(cnt_x, cnt_y, cnt_xy, total_cnt, true, false));
		}

		{

			double pr_x = 0.8;
			double pr_y = 0.75;
			double pr_xy = 0.7;

			System.out.println(pmi(pr_x, pr_y, pr_xy, true, false));
		}

		//
		// System.out.println(permutation(5, 2));

		// System.out.println(CommonMath.fastExp(0));
		// System.out.println(CommonMath.fastExp(1));
		// System.out.println(CommonMath.fastExp(-1));
		//
		// int size = 100000000;
		//
		// double[] a = ArrayMath.random(0f, 1, size);
		//
		// {
		// Timer sw1 = new Timer();
		// sw1.start();
		//
		// double sum1 = 0;
		// for (int i = 0; i < a.length; i++) {
		// sum1 += Math.exp(a[i]);
		// }
		// sw1.stop();
		//
		// Timer sw2 = new Timer();
		// sw2.start();
		//
		// double sum2 = 0;
		// for (int i = 0; i < a.length; i++) {
		// sum2 += CommonMath.fastExp(a[i]);
		// }
		// sw2.stop();
		//
		// Timer sw3 = new Timer();
		// sw3.start();
		//
		// double sum3 = 0;
		// for (int i = 0; i < a.length; i++) {
		// sum3 += CommonMath.fastExp2(a[i]);
		// }
		// sw3.stop();
		//
		// System.out.println("# exp");
		// System.out.println(sw1.toString());
		//
		// System.out.println("# fastexp");
		// System.out.println(sw2.toString());
		//
		// System.out.println("# fastexp2");
		// System.out.println(sw3.toString());
		//
		// System.out.println("# diff");
		// System.out.println(sum2 / sum1);
		//
		// System.out.println("# diff");
		// System.out.println(sum3 / sum1);
		//
		// System.out.println("# avg diff");
		// System.out.println((sum1 - sum2) / size);
		//
		// System.out.println("# avg diff");
		// System.out.println((sum1 - sum3) / size);
		// }
		//
		// {
		//
		// Timer sw1 = new Timer();
		// sw1.start();
		//
		// double sum1 = 0;
		// for (int i = 0; i < a.length; i++) {
		// sum1 += Math.log(a[i]);
		// }
		// sw1.stop();
		//
		// Timer sw2 = new Timer();
		// sw2.start();
		//
		// double sum2 = 0;
		// for (int i = 0; i < a.length; i++) {
		// sum2 += CommonMath.fastLog(a[i]);
		// }
		// sw2.stop();
		//
		// System.out.println("log");
		// System.out.println(sw1.toString());
		//
		// System.out.println("fastlog");
		// System.out.println(sw2.toString());
		//
		// System.out.println("diff");
		// System.out.println(sum2 / sum1);
		// }

		System.out.println("process ends.");
	}

	public static double max(double a, double b, double upper_bound) {
		double max = Math.max(a, b);
		if (max > upper_bound) {
			max = upper_bound;
		}
		return max;
	}

	public static int max(int a, int b, int upper_bound) {
		int max = Math.max(a, b);
		if (max > upper_bound) {
			max = upper_bound;
		}
		return max;
	}

	public static double min(double a, double b, double lower_bound) {
		double min = Math.min(a, b);
		if (min < lower_bound) {
			min = lower_bound;
		}
		return min;
	}

	public static int min(int a, int b, int lower_bound) {
		int min = Math.min(a, b);
		if (min < lower_bound) {
			min = lower_bound;
		}
		return min;
	}

	/**
	 * Maths.java in mallet
	 * 
	 * @param n
	 * @param r
	 * @return
	 */
	public static double permutation(int n, int r) {
		return Math.exp(logFactorial(n) - logFactorial(r));
	}

	/**
	 * 
	 * http://stats.stackexchange.com/questions/140935/how-does-the-logpx-y-normalize-the-pointwise-mutual-information
	 * 
	 * @param pr_x
	 * @param pr_y
	 * @param pr_xy
	 * @param normalize
	 * @return
	 */
	public static double pmi(double pr_x, double pr_y, double pr_xy, boolean natural_log, boolean normalize) {
		double ret = Double.NEGATIVE_INFINITY;

		if (Conditions.isInLeftOpenInterval(0, 1, pr_x) && Conditions.isInLeftOpenInterval(0, 1, pr_y)
				&& Conditions.isInLeftOpenInterval(0, 1, pr_xy)) {
			// ret = Math.log(pr_xy / (pr_x * pr_y));
			// ret = Math.log(pr_xy) - Math.log(pr_x) - Math.log(pr_y);
			if (natural_log) {
				ret = Math.log(pr_xy) - Math.log(pr_x) - Math.log(pr_y);
			} else {
				ret = log2(pr_xy) - log2(pr_x) - log2(pr_y);
			}
		}

		if (normalize) {
			if (ret == Double.NEGATIVE_INFINITY) {
				ret = -1;
			} else {
				if (natural_log) {
					ret /= -Math.log(pr_xy);
				} else {
					ret /= -log2(pr_xy);
				}
			}
		}
		return ret;
	}

	/**
	 * Bouma, G. (2009). Normalized (Pointwise) Mutual Information in Collocation
	 * Extraction. Proceedings of German Society for Computational Linguistics (GSCL
	 * 2009), 31–40.
	 * 
	 * https://en.wikipedia.org/wiki/Pointwise_mutual_information
	 * 
	 * @param cnt_x
	 * @param cnt_y
	 * @param cnt_xy
	 * @param size
	 * @param nature_log
	 * @param normalize
	 * @return
	 */
	public static double pmi(double cnt_x, double cnt_y, double cnt_xy, double size, boolean nature_log,
			boolean normalize) {
		return pmi(cnt_x / size, cnt_y / size, cnt_xy / size, nature_log, normalize);
	}

	/**
	 * Rectified Linear Unit
	 * 
	 * @param x
	 * @return
	 */
	public static double reLU(double x) {
		return Math.max(x, 0);
	}

	public static double reLUGradient(double x) {
		return x > 0 ? 1 : 0;
	}

	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public static double sigmoid(double x, double a, double c) {
		return 1 / (1 + Math.exp(-a * (x - c)));
	}

	public static double sigmoidGradient(double fx) {
		return (1 - fx) * fx;
	}

	public static double softSign(double x) {
		return x / (1 + Math.abs(x));
	}

	public static double softSignGradient(double x) {
		double sign = x < 0 ? -1 : 1;
		return sign / Math.pow(1 + x, 2);
	}

	public static double tanh(double x) {
		return 2 * sigmoid(2 * x) - 1;
	}

	public static double tanhGradient(double fx) {
		return 1 - fx * fx;
	}

	/**
	 * @param rank
	 *            rank of term
	 * @param M
	 *            number of distinct terms
	 * @param L
	 *            document length
	 * @return expected number of occurrences of term i in a document length L
	 */
	public static double zipfLaw(double rank, double M, double L) {
		double c = 1 / Math.log(M);
		double ret = (L * c) / rank;
		return ret;
	}

	/**
	 * Z = (x - mu) / sigma
	 * 
	 * Z ~ N(0,1)
	 * 
	 * @param x
	 * @param mu
	 * @param sigma
	 * @return
	 */
	public static double zTransform(double x, double mu, double sigma) {
		return (x - mu) / sigma;
	}

}
