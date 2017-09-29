package ohs.ml.neuralnet.nonlinearity;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;

public class ReLU implements Nonlinearity {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1725351316385413005L;

	@Override
	public void backward(DenseMatrix dY, DenseMatrix dX) {
		VectorMath.reLUGradient(dY, dX);
		
		
	}

	@Override
	public void forward(DenseMatrix X, DenseMatrix Y) {
		VectorMath.reLU(X, Y);
	}
}
