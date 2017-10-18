package ohs.ml.neuralnet.com;

import ohs.matrix.DenseMatrix;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;

public enum TaskType {
	/**
	 * X: DenseMatrix Y: IntegerArray
	 */
	CLASSIFICATION,

	/**
	 * X: IntegerMatrix Y: IntegerArray
	 */
	SEQ_CLASSIFICATION,

	/**
	 * X: IntegerMatrix Y: IntegerMatrix
	 */
	SEQ_LABELING;

	public static DenseMatrix toDenseMatrix(Object X) {
		return (DenseMatrix) X;
	}

	public static IntegerArray toIntegerArray(Object X) {
		return (IntegerArray) X;
	}

	public static IntegerMatrix toIntegerMatrix(Object X) {
		return (IntegerMatrix) X;
	}

	public boolean isValidInputType(Object X) {
		boolean ret = false;
		if (this == CLASSIFICATION && X instanceof DenseMatrix) {
			ret = true;
		} else if (this == SEQ_CLASSIFICATION && X instanceof IntegerMatrix) {
			ret = true;
		} else if (this == TaskType.SEQ_CLASSIFICATION && X instanceof IntegerMatrix) {
			ret = true;
		}
		return ret;
	}

	public boolean isValidOutputType(Object Y) {
		boolean ret = false;
		if (this == CLASSIFICATION && Y instanceof IntegerArray) {
			ret = true;
		} else if (this == SEQ_CLASSIFICATION && Y instanceof IntegerArray) {
			ret = true;
		} else if (this == TaskType.SEQ_CLASSIFICATION && Y instanceof IntegerMatrix) {
			ret = true;
		}
		return ret;
	}

	public boolean isValidType(Object X, Object Y) {
		return isValidInputType(X) && isValidOutputType(Y);
	}
}