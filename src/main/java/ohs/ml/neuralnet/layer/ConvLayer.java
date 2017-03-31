package ohs.ml.neuralnet.layer;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;

/**
 * http://cthorey.github.io./backprop_conv/
 * 
 * http://www.slideshare.net/kuwajima/cnnbp
 * 
 * http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
 * 
 * @author ohs
 *
 */
public class ConvLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -997396529874765085L;

	private int window_size;

	private int embedding_size;

	private int num_filters;

	/**
	 * filters x feature maps
	 */
	private DenseMatrix W;

	/**
	 * filter biases
	 */
	private DenseVector b;

	private DenseMatrix dW;

	private DenseVector db;

	private DenseMatrix tmp_Xn;

	private DenseMatrix tmp_Y;

	private DenseMatrix Xn;

	private DenseMatrix X;

	private DenseMatrix C;

	public ConvLayer(DenseMatrix W, DenseVector b, int window_size, int embedding_size) {
		this.W = W;
		this.b = b;
		this.window_size = window_size;
		this.embedding_size = embedding_size;
		this.num_filters = b.size();
	}

	public ConvLayer(int embedding_size, int window_size, int num_filters) {
		this.embedding_size = embedding_size;
		this.window_size = window_size;
		this.num_filters = num_filters;

		W = new DenseMatrix(num_filters, window_size * embedding_size);
		b = new DenseVector(num_filters);
	}

	@Override
	public Object backward(Object I) {
		return null;
	}

	@Override
	public DenseMatrix forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;

		this.X = X;

		int data_size = X.rowSize();
		int feat_size = embedding_size * window_size;
		// int padding_size = window_size - 1;
		// int new_data_size = data_size + 2 * padding_size;

		if (tmp_Xn == null || tmp_Xn.rowSize() < data_size) {
			tmp_Xn = new DenseMatrix(data_size, feat_size);
		}

		Xn = tmp_Xn.rowsAsMatrix(data_size);
		Xn.setAll(0);

		for (int i = 0; i < data_size; i++) {
			DenseVector xn = Xn.row(i);
			int start = i - window_size + 1;
			int end = i;

			for (int j = start, u = 0; j <= end; j++) {
				if (j < 0) {
					u += embedding_size;
				} else {
					DenseVector x = X.row(j);

					for (int k = 0; k < x.size(); k++) {
						xn.set(u++, x.value(k));
					}
				}
			}
			xn.summation();
		}

		System.out.println(Xn);

		// for (int i = 0; i < X.rowSize(); i++) {
		// int j = i + padding_size;
		// VectorUtils.copy(X.row(i), Xn.row(j));
		// }

		// int feat_map_size = Xn.rowSize() - window_size + 1;
		//
		// if (tmp_Y == null || tmp_Y.colSize() < feat_map_size) {
		// tmp_Y = new DenseMatrix(num_filters, feat_map_size);
		// }
		//
		// DenseMatrix Y = tmp_Y.rowsAsMatrix(num_filters);
		// DenseVector e = new DenseVector(W.colSize());
		//
		// for (int i = 0; i < Xn.rowSize() - window_size; i++) {
		// VectorUtils.copyRows(Xn, i, i + window_size, e);
		//
		// for (int j = 0; j < W.rowSize(); j++) {
		// /*
		// * w: filter
		// */
		// DenseVector w = W.row(j);
		// double c = VectorMath.dotProduct(w, e) + b.value(j);
		// Y.set(j, i, c);
		//
		// }
		// }
		return null;
	}

	@Override
	public DenseMatrix getB() {
		return b.toDenseMatrix();
	}

	@Override
	public DenseMatrix getDB() {
		return db.toDenseMatrix();
	}

	@Override
	public DenseMatrix getDW() {
		return dW;
	}

	public int getEmbeddingSize() {
		return embedding_size;
	}

	@Override
	public int getInputSize() {
		return W.rowSize();
	}

	@Override
	public int getOutputSize() {
		return W.colSize();
	}

	@Override
	public DenseMatrix getW() {
		return W;
	}

	@Override
	public void init() {
		if (is_testing) {

		} else {
			VectorMath.random(0, 1, W);
			W.add(-0.5);
			W.multiply(1f / W.colSize());

			dW = W.copy(true);
			db = b.copy(true);
		}

	}

}
