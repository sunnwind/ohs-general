package ohs.ml.neuralnet.layer;

import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;

public class MaxPoolingLayer extends Layer {

	private static final long serialVersionUID = -5068216034849402227L;

	/**
	 * data x filters
	 */
	private IntegerMatrix L;

	private int num_filters;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private IntegerMatrix tmp_L = new IntegerMatrix(new int[0][0]);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	/**
	 * data x filter maps x filters
	 */
	private DenseTensor X;

	/**
	 * data x filters
	 */
	private DenseMatrix Y;

	public MaxPoolingLayer(int num_filters) {
		this.num_filters = num_filters;
	}

	@Override
	public Object backward(Object I) {

		/*
		 * data size x filters
		 */
		DenseMatrix dY = (DenseMatrix) I;

		int len = X.sizeOfInnerVectors();

		if (tmp_dX.rowSize() < len) {
			tmp_dX = new DenseMatrix(len, num_filters);
		}

		int data_size = dY.rowSize();

		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(data_size);

		for (int u = 0, start = 0; u < data_size; u++) {
			DenseVector dYm = dY.row(u);
			DenseMatrix Xm = X.row(u);
			IntegerArray Lm = L.get(u);

			int num_filters = Lm.size();
			int num_feature_maps = Xm.rowSize();

			DenseMatrix dXm = tmp_dX.rows(start, num_feature_maps);
			dXm.setAll(0);

			for (int filter_idx = 0; filter_idx < num_filters; filter_idx++) {
				int max_feat_map_idx = Lm.get(filter_idx);
				double v = 0;
				v = Xm.value(max_feat_map_idx, filter_idx) * dYm.value(filter_idx);
				dXm.add(max_feat_map_idx, filter_idx, v);
			}

			dX.add(dXm);

			start += num_feature_maps;
		}

		// DenseMatrix dX = tmp_dX.rows(num_feature_maps);
		// dX.setAll(0);

		// for (int j = 0; j < num_filters; j++) {
		// int i = L.value(j);
		// dX.add(i, j, dY.value(i, j));
		// }

		// for (int i = 0; i < num_feature_maps; i++) {
		// DenseVector x = X.row(i);
		// DenseVector dx = dX.row(i);
		// int max_idx = L.get(i);
		// double dy = dY.row(0).value(max_idx);
		// dx.add(max_idx, dy * x.value(max_idx));
		// }
		return dX;
	}

	@Override
	public Layer copy() {
		return new MaxPoolingLayer(num_filters);
	}

	@Override
	public Object forward(Object I) {
		/*
		 * X = date x feature maps x filters
		 */
		DenseTensor X = (DenseTensor) I;
		this.X = X;
		int data_size = X.size();

		if (tmp_L.size() < data_size) {
			tmp_L = new IntegerMatrix(data_size, num_filters);
			tmp_Y = new DenseMatrix(data_size, num_filters);
		}

		L = tmp_L.subMatrix(data_size);
		Y = tmp_Y.rows(data_size);

		Y.setAll(-Double.MAX_VALUE);

		for (int u = 0; u < data_size; u++) {

			/*
			 * Xm = feature maps x filters
			 */
			DenseMatrix Xm = X.get(u);
			DenseVector Ym = Y.get(u);
			IntegerArray Lm = L.get(u);

			int num_feature_maps = Xm.rowSize();
			for (int feat_map_idx = 0; feat_map_idx < num_feature_maps; feat_map_idx++) {
				DenseVector x = Xm.row(feat_map_idx);
				for (int filter_idx = 0; filter_idx < num_filters; filter_idx++) {
					double feat_map_val = x.value(filter_idx);
					double max_feat_map_val = Ym.value(filter_idx);
					if (feat_map_val > max_feat_map_val) {
						Lm.set(filter_idx, feat_map_idx);
						Ym.set(filter_idx, feat_map_val);
						max_feat_map_val = feat_map_val;
					}
				}
			}
		}

		if (Y.sum() == 0) {
			System.out.println();
		}

		return Y;
	}

	@Override
	public int getOutputSize() {
		return num_filters;
	}

}
