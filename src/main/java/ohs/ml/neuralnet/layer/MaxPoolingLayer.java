package ohs.ml.neuralnet.layer;

import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;

public class MaxPoolingLayer extends Layer {

	private static final long serialVersionUID = -5068216034849402227L;

	/**
	 * data x filters
	 */
	private DenseTensor L;

	private int num_filters;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_L = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	/**
	 * data x filter maps x filters
	 */
	private DenseTensor X;

	/**
	 * data x filters
	 */
	private DenseTensor Y;

	public MaxPoolingLayer(int num_filters) {
		this.num_filters = num_filters;
	}

	@Override
	public DenseTensor backward(Object I) {

		/*
		 * data size x filters
		 */
		DenseTensor dY = (DenseTensor) I;
		DenseTensor X = this.X;
		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(dY.size());

		VectorUtils.enlarge(tmp_dX, X.sizeOfInnerVectors(), num_filters);

		for (int i = 0, start = 0; i < dY.size(); i++) {
			DenseMatrix dYm = dY.row(i);
			DenseMatrix Xm = X.row(i);
			DenseMatrix Lm = L.row(i);

			int num_feature_maps = Xm.rowSize();

			DenseMatrix dXm = tmp_dX.subMatrix(start, num_feature_maps);
			dXm.setAll(0);

			start += num_feature_maps;

			for (int filter_idx = 0; filter_idx < num_filters; filter_idx++) {
				int max_feat_map_idx = (int) Lm.value(0, filter_idx);
				double max_feat_map_val = Xm.value(max_feat_map_idx, filter_idx);
				double v = max_feat_map_val * dYm.value(0, filter_idx);
				dXm.add(max_feat_map_idx, filter_idx, v);
			}

			dX.add(dXm);
		}

		return dX;
	}

	// public Object backwardOld(Object I) {
	//
	// /*
	// * data size x filters
	// */
	// DenseMatrix dY = (DenseMatrix) I;
	//
	// int len = X.sizeOfInnerVectors();
	//
	// if (tmp_dX.rowSize() < len) {
	// tmp_dX = new DenseMatrix(len, num_filters);
	// }
	//
	// int data_size = dY.rowSize();
	//
	// DenseTensor dX = new DenseTensor();
	// dX.ensureCapacity(data_size);
	//
	// for (int u = 0, start = 0; u < data_size; u++) {
	// DenseVector dYm = dY.row(u);
	// DenseMatrix Xm = X.row(u);
	// IntegerArray Lm = L.get(u);
	//
	// int num_filters = Lm.size();
	// int num_feature_maps = Xm.rowSize();
	//
	// DenseMatrix dXm = tmp_dX.subMatrix(start, num_feature_maps);
	// dXm.setAll(0);
	//
	// for (int filter_idx = 0; filter_idx < num_filters; filter_idx++) {
	// int max_feat_map_idx = Lm.get(filter_idx);
	// double v = 0;
	// v = Xm.value(max_feat_map_idx, filter_idx) * dYm.value(filter_idx);
	// dXm.add(max_feat_map_idx, filter_idx, v);
	// }
	//
	// dX.add(dXm);
	//
	// start += num_feature_maps;
	// }
	//
	// return dX;
	// }

	@Override
	public MaxPoolingLayer copy() {
		return new MaxPoolingLayer(num_filters);
	}

	@Override
	public DenseTensor forward(Object I) {
		this.X = (DenseTensor) I;

		/*
		 * X = date x feature maps x filters
		 * 
		 * L = filter index => max_feat_map_idx
		 * 
		 * Y = filter index => max_feat_value
		 */
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		DenseTensor L = new DenseTensor();

		Y.ensureCapacity(X.size());
		L.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_L, X.size(), num_filters);
		VectorUtils.enlarge(tmp_Y, X.size(), num_filters);

		int start = 0;

		for (int i = 0; i < X.size(); i++) {

			/*
			 * Xm = feature maps x filters
			 */
			DenseMatrix Xm = X.get(i);
			DenseMatrix Ym = tmp_Y.subMatrix(i, 1);
			DenseMatrix Lm = tmp_L.subMatrix(i, 1);

			Ym.setAll(-Double.MAX_VALUE);
			Lm.setAll(0);

			int num_feat_maps = Xm.rowSize();
			for (int feat_map_idx = 0; feat_map_idx < num_feat_maps; feat_map_idx++) {
				DenseVector filters = Xm.row(feat_map_idx);

				for (int filter_idx = 0; filter_idx < num_filters; filter_idx++) {
					double feat_map_val = filters.value(filter_idx);
					double max_feat_map_val = Ym.value(0, filter_idx);

					if (feat_map_val > max_feat_map_val) {
						Lm.set(0, filter_idx, feat_map_idx);
						Ym.set(0, filter_idx, feat_map_val);

						max_feat_map_val = feat_map_val;
					}
				}
			}

			Ym.sumRows();
			Lm.sumRows();

			Y.add(Ym);
			L.add(Lm);
		}

		this.Y = Y;
		this.L = L;

		return Y;
	}

	@Override
	public int getOutputSize() {
		return num_filters;
	}

}
