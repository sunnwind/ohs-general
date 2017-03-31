package ohs.ml.neuralnet.nonlinearity;

import java.io.Serializable;

import ohs.matrix.DenseMatrix;

public interface Nonlinearity extends Serializable {

	public void backward(DenseMatrix dY, DenseMatrix dX);

	public void forward(DenseMatrix X, DenseMatrix Y);

}
