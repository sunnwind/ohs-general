package ohs.ml.neuralnet.layer;

import java.util.List;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
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

	private int window_size;

	private int embedding_size;

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

	private int padding_size = 0;

	private int concat_embedding_size = 0;

	private boolean use_padding = false;

	private DenseMatrix P;

	public ConvLayer(DenseMatrix W, DenseVector b, int window_size, int embedding_size) {
		this.W = W;
		this.b = b;
		this.window_size = window_size;
		this.embedding_size = embedding_size;
		padding_size = window_size - 1;
		concat_embedding_size = embedding_size * window_size;

		P = new DenseMatrix(padding_size, embedding_size);
	}

	public ConvLayer(int embedding_size, int window_size, int filter_size) {
		this(new DenseMatrix(filter_size, window_size * embedding_size), new DenseVector(filter_size), window_size, embedding_size);
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
		int padded_data_size = data_size + padding_size;

		DenseMatrix Xp = null;

		{
			List<DenseVector> l = Generics.newArrayList(padded_data_size);

			for (DenseVector p : P) {
				l.add(p);
			}

			for (DenseVector x : X) {
				l.add(x);
			}

			Xp = new DenseMatrix(l);
		}

		if (tmp_Xc == null || tmp_Xc.rowSize() < data_size) {
			tmp_Xc = new DenseMatrix(data_size, concat_embedding_size);
		}
		
		/*
		 * Xc = data x concatenated embeddings
		 */

		Xc = tmp_Xc.rowsAsMatrix(data_size);
		Xc.setAll(0);

		for (int i = 0; i < data_size; i++) {
			int start = i;
			int end = i + window_size;

			DenseVector xc = Xc.row(i);

			for (int j = start, k = 0; j < end; j++) {
				DenseVector x = Xp.row(j);
				for (int l = 0; l < x.size(); l++) {
					xc.add(k++, x.value(l));
				}
			}
		}

		/*
		 * C = filter x concatenated embeddings
		 */

		DenseMatrix C = new DenseMatrix(W.rowSize(), Xc.rowSize());
		VectorMath.productColumns(W, Xc, C, false);

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

	public void usePadding(boolean use_padding) {
		this.use_padding = use_padding;
	}

}
