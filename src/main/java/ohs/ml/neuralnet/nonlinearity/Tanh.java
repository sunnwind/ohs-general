package ohs.ml.neuralnet.nonlinearity;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;

public class Tanh implements Nonlinearity {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1521093436132810767L;

	@Override
	public void backward(DenseMatrix dY, DenseMatrix dX) {
		VectorMath.tanhGradient(dY, dX);
	}

	@Override
	public void forward(DenseMatrix X, DenseMatrix Y) {
		VectorMath.tanh(X, Y);
	}

}
