package ohs.ml.neuralnet.nonlinearity;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;

public class Sigmoid implements Nonlinearity {

	@Override
	public void backward(DenseMatrix dY, DenseMatrix dX) {
		VectorMath.sigmoidGradient(dY, dX);
	}

	@Override
	public void forward(DenseMatrix X, DenseMatrix Y) {
		VectorMath.sigmoid(X, Y);
	}

}
