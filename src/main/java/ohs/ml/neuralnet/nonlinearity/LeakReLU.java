package ohs.ml.neuralnet.nonlinearity;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;

public class LeakReLU implements Nonlinearity {

	private double k = 0.01;

	@Override
	public void backward(DenseMatrix A, DenseMatrix dA) {
		VectorMath.leakReLUGradient(A, k, dA);
	}

	@Override
	public void forward(DenseMatrix Z, DenseMatrix A) {
		VectorMath.leakReLU(Z, k, A);
	}

}
