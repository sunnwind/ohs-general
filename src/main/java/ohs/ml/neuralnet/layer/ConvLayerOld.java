package ohs.ml.neuralnet.layer;

import java.util.List;

import ohs.math.VectorMath;
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
public class ConvLayerOld extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -997396529874765085L;

	private int window_size;

	private int emb_size;

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

	private DenseMatrix tmp_Xc;

	private DenseMatrix Xc;

	private DenseMatrix fwd_X;

	private DenseMatrix C;

	private int pad_size = 0;

	private int filter_size = 0;

	private boolean use_padding = false;

	private DenseMatrix P;

	public int getWindowSize() {
		return window_size;
	}

	public int getFilterSize() {
		return W.rowSize();
	}

	public ConvLayerOld(DenseMatrix W, DenseVector b, int window_size, int embed_size) {
		this.W = W;
		this.b = b;
		this.window_size = window_size;
		this.emb_size = embed_size;
		pad_size = window_size - 1;
		filter_size = embed_size * window_size;

		P = new DenseMatrix(1, embed_size);
	}

	public ConvLayerOld(int emb_size, int window_size, int num_filters) {
		this(new DenseMatrix(num_filters, window_size * emb_size), new DenseVector(num_filters), window_size, emb_size);
	}

	@Override
	public Object backward(Object I) {
		return null;
	}

	@Override
	public DenseMatrix forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;

		this.fwd_X = X;

		int data_size = X.rowSize();
		int feat_map_size = data_size - window_size + 1;

		DenseMatrix Xp = null;

		if (tmp_Xc == null || tmp_Xc.rowSize() < feat_map_size) {
			tmp_Xc = new DenseMatrix(feat_map_size, filter_size);
		}

		/*
		 * Xc = feature_map_size x concatenated embeddings
		 */

		Xc = tmp_Xc.rows(feat_map_size);
		Xc.setAll(0);

		for (int i = 0; i < feat_map_size; i++) {
			int start = i;
			int end = i + window_size;

			DenseVector xc = Xc.row(i);

			for (int j = start, loc = 0; j < end; j++) {
				DenseVector x = X.row(j);
				for (int k = 0; k < x.size(); k++) {
					xc.add(loc++, x.value(k));
				}
			}
		}

		/*
		 * C = filter x concatenated embeddings
		 */

		DenseMatrix C = new DenseMatrix(W.rowSize(), Xc.rowSize());
		VectorMath.productRows(W, Xc, C, false);

		for (int i = 0; i < C.rowSize(); i++) {
			DenseVector c = C.row(i);
			double bias = b.value(i);
			for (int j = 0; j < c.size(); j++) {
				c.add(j, bias);
			}
		}

		return C;
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
	public void init() {
		ParameterInitializer.init2(W);
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
	public void prepare() {
		dW = W.copy(true);
		db = b.copy(true);
	}

	public void usePadding(boolean use_padding) {
		this.use_padding = use_padding;
	}

}
