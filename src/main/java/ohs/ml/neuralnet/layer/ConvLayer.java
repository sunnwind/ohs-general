package ohs.ml.neuralnet.layer;

import java.util.List;

import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
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
 * @author ohs
 *
 */
public class ConvLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -997396529874765085L;

	/**
	 * filter biases
	 */
	private DenseVector B;

	private DenseVector db;

	private DenseMatrix dW;

	private int emb_size;

	private int filter_size = 0;

	private DenseMatrix fwd_X;

	private DenseMatrix P;

	private int pad_size = 0;

	private DenseMatrix T;

	private DenseMatrix tmp_Xc;

	private boolean use_padding = false;

	/**
	 * filters x filter size
	 */
	private DenseMatrix W;

	private int window_size;

	private DenseMatrix Xc;

	public ConvLayer(DenseMatrix W, DenseVector b, int window_size, int embed_size) {
		this.W = W;
		this.B = b;
		this.window_size = window_size;
		this.emb_size = embed_size;
		pad_size = window_size - 1;
		filter_size = embed_size * window_size;

		P = new DenseMatrix(1, embed_size);
	}

	public ConvLayer(int emb_size, int window_size, int num_filters) {
		this(new DenseMatrix(num_filters, window_size * emb_size), new DenseVector(num_filters), window_size, emb_size);
	}

	@Override
	public Object backward(Object I) {
		return null;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * T: filters X feature maps
	 * 
	 * @see ohs.ml.neuralnet.layer.Layer#forward(java.lang.Object)
	 */
	@Override
	public DenseMatrix forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;

		this.fwd_X = X;

		int data_size = X.rowSize();
		int num_feature_maps = data_size - window_size + 1;
		int num_filters = W.rowSize();

		if (tmp_Xc == null || tmp_Xc.rowSize() < num_feature_maps) {
			tmp_Xc = new DenseMatrix(num_feature_maps, filter_size);
		}

		/*
		 * Xc = feature maps x filter size (size of concatenated embeddings)
		 */

		Xc = tmp_Xc.rows(num_feature_maps);
		Xc.setAll(0);

		for (int i = 0; i < num_feature_maps; i++) {
			DenseVector xc = Xc.row(i);
			
			try {
				VectorUtils.copyRows(X, i, i + window_size, xc);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		/*
		 * T = filters X feature maps
		 */

		T = new DenseMatrix(num_filters, num_feature_maps);

		VectorMath.productRows(W, Xc, T, false);

		for (int i = 0; i < T.rowSize(); i++) {
			DenseVector t = T.row(i);
			t.add(B.value(i));

		}

		return T;
	}

	@Override
	public DenseMatrix getB() {
		return B.toDenseMatrix();
	}

	@Override
	public DenseMatrix getDB() {
		return db.toDenseMatrix();
	}

	@Override
	public DenseMatrix getDW() {
		return dW;
	}

	@Override
	public int getInputSize() {
		return emb_size;
	}

	@Override
	public int getOutputSize() {
		return filter_size;
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
		db = B.copy(true);
	}

	public void usePadding(boolean use_padding) {
		this.use_padding = use_padding;
	}

}
