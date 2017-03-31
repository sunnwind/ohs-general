package ohs.ml.neuralnet.layer;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.types.number.IntegerArray;

public class MaxPoolingLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5068216034849402227L;

	private DenseMatrix Y;

	private DenseMatrix X;

	private IntegerArray maxIndexes;

	private DenseMatrix tmp_dX;

	public MaxPoolingLayer(int num_filters) {
		Y = new DenseMatrix(num_filters);
		maxIndexes = new IntegerArray(num_filters);
	}

	@Override
	public Object backward(Object I) {
		DenseMatrix dY = (DenseMatrix) I;
		int data_size = dY.rowSize();

		if (tmp_dX == null || tmp_dX.rowSize() < data_size) {
			tmp_dX = new DenseMatrix(data_size, X.colSize());
		}

		DenseMatrix dX = tmp_dX.rowsAsMatrix(data_size);
		dX.setAll(0);

		for (int i = 0; i < X.rowSize(); i++) {
			DenseVector dx = dX.row(i);
			DenseVector dy = dY.row(i);
			int max_idx = maxIndexes.get(i);

			dx.addAt(max_idx, dy.value(max_idx));
		}
		return dX;
	}

	@Override
	public Object forward(Object I) {
		/*
		 * X = num_filters x feature map
		 */
		DenseMatrix X = (DenseMatrix) I;
		this.X = X;

		Y.setAll(0);
		maxIndexes.clear();

		for (int i = 0; i < X.rowSize(); i++) {
			DenseVector x = X.row(i);
			DenseVector y = Y.row(i);
			int max_idx = x.argMax();
			double max_value = x.value(max_idx);

			maxIndexes.add(max_idx);
			y.add(max_idx, max_value);
		}
		return Y;
	}

	@Override
	public DenseMatrix getB() {
		return null;
	}

	@Override
	public DenseMatrix getDB() {
		return null;
	}

	@Override
	public DenseMatrix getDW() {
		return null;
	}

	@Override
	public int getInputSize() {
		return 0;
	}

	@Override
	public int getOutputSize() {
		return 0;
	}

	@Override
	public DenseMatrix getW() {
		return null;
	}

	@Override
	public void init() {

	}

}
