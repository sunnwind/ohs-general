package ohs.ml.neuralnet.layer;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.types.number.IntegerArray;

public class MaxPoolingLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5068216034849402227L;

	private IntegerArray L;

	private DenseMatrix tmp_dX;

	private DenseMatrix X;

	private DenseVector Y;

	public MaxPoolingLayer(int num_filters) {
		Y = new DenseVector(num_filters);
		L = new IntegerArray(new int[num_filters]);
	}

	@Override
	public Object backward(Object I) {
		DenseMatrix dY = (DenseMatrix) I;

		int data_size = X.rowSize();

		if (tmp_dX == null || tmp_dX.rowSize() < data_size) {
			tmp_dX = new DenseMatrix(data_size, X.colSize());
		}

		DenseMatrix dX = tmp_dX.rows(data_size);
		dX.setAll(0);

		for (int i = 0; i < X.rowSize(); i++) {
			DenseVector x = X.row(i);
			DenseVector dx = dX.row(i);
			int max_idx = L.get(i);
			double dy = dY.row(0).value(max_idx);
			dx.add(max_idx, dy * x.value(max_idx));
		}
		return dX;
	}

	@Override
	public Object forward(Object I) {
		/*
		 * X = filters x feature maps
		 */
		DenseMatrix X = (DenseMatrix) I;
		this.X = X;

		Y.setAll(0);

		for (int i = 0; i < X.rowSize(); i++) {
			DenseVector x = X.row(i);
			int max_j = x.argMax();
			double max = x.value(max_j);

			L.set(i, max_j);
			Y.add(i, max);
		}

		return new DenseMatrix(new DenseVector[] { Y });
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
