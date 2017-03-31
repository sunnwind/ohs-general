package ohs.math.optimize;

import ohs.math.CommonMath;

public class Sigmoid implements Function {

	@Override
	public double gradient(double a) {
		return CommonMath.sigmoidGradient(a);
	}

	@Override
	public double gradient(double[] a, double[] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			b[i] = CommonMath.sigmoidGradient(a[i]);
			sum += b[i];
		}
		return sum;
	}

	@Override
	public double gradient(double[][] a, double[][] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			sum += gradient(a[i], b[i]);
		}
		return sum;
	}

	@Override
	public double value(double a) {
		return CommonMath.sigmoid(a);
	}

	@Override
	public double value(double[] a, double[] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			b[i] = value(a[i]);
			sum += b[i];
		}
		return sum;
	}

	@Override
	public double value(double[][] a, double[][] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			sum += value(a[i], b[i]);
		}
		return sum;
	}

}
