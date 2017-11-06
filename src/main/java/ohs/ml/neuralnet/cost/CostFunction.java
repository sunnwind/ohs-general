package ohs.ml.neuralnet.cost;

import java.io.Serializable;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.types.number.IntegerArray;

public interface CostFunction extends Serializable {

	public DenseMatrix evaluate(DenseMatrix Yh, DenseVector y);

	public double getCost();

	public int getCorrectCount();
}
