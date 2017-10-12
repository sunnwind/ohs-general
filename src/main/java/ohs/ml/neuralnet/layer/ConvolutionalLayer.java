package ohs.ml.neuralnet.layer;

import java.util.List;

import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.utils.Generics;

/**
 * http://cthorey.github.io./backprop_conv/
 * 
 * http://www.slideshare.net/kuwajima/cnnbp
 * 
 * http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
 * 
 * https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/
 * 
 * @author ohs
 *
 */
public class ConvolutionalLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -997396529874765085L;

	/**
	 * filter biases
	 */
	private DenseVector b;

	private DenseVector db;

	private DenseMatrix dW;

	private int emb_size;

	/**
	 * embedding size x window size
	 */
	private int filter_size;

	private int num_filters;

	private int pad_size;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_dZ = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private DenseMatrix tmp_Z = new DenseMatrix(0);

	/**
	 * filters x filter size (embedding size x window size)
	 */
	private DenseMatrix W;

	private int window_size;

	/**
	 * data x words x embeddings
	 */
	private DenseTensor X;

	/**
	 * data x feature maps x filters
	 */
	private DenseTensor Y;

	/**
	 * data x feature maps x filters
	 */
	private DenseTensor Z;

	/**
	 * @param W
	 *            feature maps x filters
	 * @param b
	 * 
	 * @param window_size
	 * @param emb_size
	 */
	public ConvolutionalLayer(DenseMatrix W, DenseVector b, int window_size, int emb_size) {
		this.W = W;
		this.b = b;

		this.window_size = window_size;
		this.emb_size = emb_size;

		filter_size = W.rowSize();
		num_filters = W.colSize();

		pad_size = window_size - 1;
	}

	public ConvolutionalLayer(int emb_size, int window_size, int num_filters) {
		this(new DenseMatrix(window_size * emb_size, num_filters), new DenseVector(num_filters), window_size, emb_size);
	}

	@Override
	public DenseTensor backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		int data_size = dY.size();

		VectorUtils.enlarge(tmp_dZ, dY.sizeOfInnerVectors(), filter_size);
		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), emb_size);

		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(data_size);

		for (int i = 0, start = 0; i < data_size; i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix Zm = Z.get(i);
			DenseMatrix Ym = Y.get(i);
			DenseMatrix dYm = dY.get(i);

			int len = dYm.rowSize();

			DenseMatrix dZm = tmp_dZ.rows(start, len);
			DenseMatrix dXm = tmp_dX.rows(start, len);

			dZm.setAll(0);
			dXm.setAll(0);

			for (int j = 0; j < len; j++) {
				VectorMath.outerProduct(Zm.row(j), dYm.row(j), dW, true);
				VectorMath.add(dYm.row(j), db);
			}

			VectorMath.productRows(dYm, W, dZm, false);

			for (int j = 0; j < len; j++) {
				DenseVector dx = dXm.row(j);
				DenseVector dz = dZm.row(j);

				/*
				 * copy value from k-th at a filter to l-th at an embedding
				 */

				for (int k = 0; k < filter_size; k++) {
					int l = k % emb_size;
					dx.add(l, dz.value(k));
				}
			}

			dXm.multiply(1d / (window_size * len));

			dX.add(dXm);

			start += len;

		}

		return dX;
	}

	@Override
	public Layer copy() {
		return new ConvolutionalLayer(W, b, window_size, emb_size);
	}

	/*
	 * I = data size x feature maps x filters
	 */
	@Override
	public DenseTensor forward(Object I) {
		DenseTensor X = (DenseTensor) I;
		this.X = X;

		int data_size = X.rowSize();

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), num_filters);
		VectorUtils.enlarge(tmp_Z, X.sizeOfInnerVectors(), filter_size);

		Y = new DenseTensor();
		Y.ensureCapacity(data_size);

		Z = new DenseTensor();
		Z.ensureCapacity(data_size);

		for (int i = 0, start = 0; i < data_size; i++) {
			DenseMatrix Xm = X.get(i);
			int len = Xm.rowSize();

			/*
			 * sentence length = # of feature maps
			 */
			int num_feature_maps = len;

			tmp_Z.rows(start, num_feature_maps);

			/*
			 * Zm = feature maps x filter size (concatenated embeddings)
			 */

			DenseMatrix Zm = tmp_Z.rows(start, num_feature_maps);
			Zm.setAll(0);

			/*
			 * Xm (sentence length, embedding size) => Zm (feature maps, embedding size *
			 * window size)
			 */

			for (int j = 0; j < num_feature_maps; j++) {
				DenseVector z = Zm.row(j);
				for (int k = j, l = 0; k < j + window_size && k < num_feature_maps; k++) {
					DenseVector x = Xm.row(k);
					ArrayUtils.copy(x.values(), 0, z.values(), l, x.size());
					l += x.size();
				}
			}

			Z.add(Zm);

			/*
			 * Ym = feature maps X filters
			 */

			DenseMatrix Ym = tmp_Y.rows(start, num_feature_maps);

			VectorMath.product(Zm, W, Ym, false);

			VectorMath.add(Ym, b, Ym);

			Y.add(Ym);

			start += data_size;

		}

		return Y;
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
		return emb_size;
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

	public int getWindowSize() {
		return window_size;
	}

	@Override
	public void init() {
		ParameterInitializer.init2(W);
	}

	@Override
	public void prepare() {
		dW = W.copy(true);
		db = b.copy(true);
	}

}
