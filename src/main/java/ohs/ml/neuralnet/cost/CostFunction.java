package ohs.ml.neuralnet.cost;

import java.io.Serializable;

import ohs.matrix.DenseMatrix;
import ohs.types.number.IntegerArray;

public interface CostFunction extends Serializable {

	public DenseMatrix evaluate(DenseMatrix Yh, IntegerArray y);

	public double getCost();

	public int getCorrectCnt();
}
