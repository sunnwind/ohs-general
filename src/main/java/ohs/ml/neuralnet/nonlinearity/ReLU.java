package ohs.ml.neuralnet.nonlinearity;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;

public class ReLU implements Nonlinearity {

	@Override
	public void backward(DenseMatrix dY, DenseMatrix dX) {
		VectorMath.reLUGradient(dY, dX);
	}

	@Override
	public void forward(DenseMatrix X, DenseMatrix Y) {
		VectorMath.reLU(X, Y);
	}

}
