package ohs.ml.neuralnet.layer;

import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.types.number.IntegerArray;

public class MultiWindowMaxPoolingLayer extends Layer {

	private static final long serialVersionUID = -5068216034849402227L;

	/**
	 * data x filters
	 */
	private DenseTensor L;

	private int num_filters;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_L = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private IntegerArray windowSizes;

	/**
	 * data x filter maps x filters
	 */
	private DenseTensor X;

	/**
	 * data x filters
	 */
	private DenseTensor Y;

	public MultiWindowMaxPoolingLayer(int num_filters, int[] window_sizes) {
		this.num_filters = num_filters;
		this.windowSizes = new IntegerArray(window_sizes);
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

			DenseMatrix dXm = tmp_dX.subMatrix(start, Xm.rowSize());
			dXm.setAll(0);

			start += Xm.rowSize();

			int num_feat_maps = Xm.rowSize() / windowSizes.size();
			int num_new_filters = num_filters * windowSizes.size();

			for (int new_filter_idx = 0; new_filter_idx < num_new_filters; new_filter_idx++) {
				int max_new_feat_map_idx = (int) Lm.value(0, new_filter_idx);

				int win = new_filter_idx / num_filters;
				int feat_map_idx = max_new_feat_map_idx - win * num_feat_maps;
				int filter_idx = new_filter_idx - win * num_filters;

				double max_feat_map_val = Xm.value(feat_map_idx, filter_idx);
				double v = max_feat_map_val * dYm.value(0, new_filter_idx);

				dXm.add(feat_map_idx, filter_idx, v);
			}

			for (int filter_idx = 0; filter_idx < num_filters; filter_idx++) {
				int max_feat_map_idx = (int) Lm.value(0, filter_idx);
				double max_feat_map_val = Xm.value(max_feat_map_idx, filter_idx);
				double v = max_feat_map_val * dYm.value(0, filter_idx);
				dXm.add(max_feat_map_idx, filter_idx, v);
			}

			dX.add(dXm);
		}

		return dX;
	};

	@Override
	public MultiWindowMaxPoolingLayer copy() {
		return new MultiWindowMaxPoolingLayer(num_filters, windowSizes.values());
	}

	@Override
	public DenseTensor forward(Object I) {
		this.X = (DenseTensor) I;

		/*
		 * X = data x feature maps x filters
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

		int output_size = num_filters * windowSizes.size();

		VectorUtils.enlarge(tmp_L, X.size(), output_size);
		VectorUtils.enlarge(tmp_Y, X.size(), output_size);

		int start = 0;

		for (int i = 0; i < X.size(); i++) {

			/*
			 * Xm = feature maps x filters (feature maps => original feature maps x window
			 * sizes)
			 */
			DenseMatrix Xm = X.get(i);
			DenseMatrix Ym = tmp_Y.subMatrix(i, 1);
			DenseMatrix Lm = tmp_L.subMatrix(i, 1);

			Ym.setAll(-Double.MAX_VALUE);
			Lm.setAll(0);

			for (int win = 0; win < windowSizes.size(); win++) {
				int num_feat_maps = Xm.rowSize() / windowSizes.size();

				for (int feat_map_idx = 0; feat_map_idx < num_feat_maps; feat_map_idx++) {
					int new_feat_map_idx = win * num_feat_maps + feat_map_idx;
					DenseVector filters = Xm.row(new_feat_map_idx);

					for (int filter_idx = 0; filter_idx < num_filters; filter_idx++) {
						int new_filter_idx = win * num_filters + filter_idx;
						double feat_map_val = filters.value(filter_idx);
						double max_feat_map_val = Ym.value(0, new_filter_idx);

						if (feat_map_val > max_feat_map_val) {
							Lm.set(0, new_filter_idx, new_feat_map_idx);
							Ym.set(0, new_filter_idx, feat_map_val);
							max_feat_map_val = feat_map_val;
						}
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
	public int getInputSize() {
		return num_filters;
	}

	@Override
	public int getOutputSize() {
		return num_filters * windowSizes.size();
	}

}
