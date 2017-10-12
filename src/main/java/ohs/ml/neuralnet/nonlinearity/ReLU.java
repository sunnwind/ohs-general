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
		// for (int i = 0; i < X.rowSize(); i++) {
		// DenseVector x = X.row(i);
		// DenseVector y = Y.row(i);
		//
		// for (int j = 0; j < x.size(); j++) {
		// double v1 = x.value(j);
		// double v2 = CommonMath.reLU(v1);
		// y.add(j, v2);
		// }
		// }
		VectorMath.reLU(X, Y); 
	}
}
