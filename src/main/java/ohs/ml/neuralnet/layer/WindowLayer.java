package ohs.ml.neuralnet.layer;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;

/**
 * http://cthorey.github.io./backprop_conv/
 * 
 * http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
 * 
 * @author ohs
 *
 */
public class WindowLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7566013299483034685L;

	private int window_size;

	private int embedding_size;

	private DenseMatrix tmp_Y;

	private int output_size;

	private DenseMatrix tmp_dX;

	public WindowLayer(int window_size, int embedding_size) {
		this.window_size = window_size;
		this.embedding_size = embedding_size;

		output_size = (2 * window_size + 1) * embedding_size;
	}

	@Override
	public Object backward(Object I) {
		DenseMatrix dY = (DenseMatrix) I;

		int data_size = dY.rowSize();

		if (tmp_dX == null || tmp_dX.rowSize() < data_size) {
			tmp_dX = new DenseMatrix(data_size, embedding_size);
		}

		DenseMatrix dX = tmp_dX.rows(data_size);

		for (int i = 0; i < data_size; i++) {
			
		}

		return null;
	}

	@Override
	public DenseMatrix forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;

		if (tmp_Y == null || tmp_Y.rowSize() < X.rowSize()) {
			tmp_Y = new DenseMatrix(X.rowSize(), output_size);
		}

		DenseMatrix Y = tmp_Y.rows(X.rowSize());

		for (int i = 0; i < X.rowSize(); i++) {
			DenseVector y = Y.row(i);

			for (int j = i - window_size, u = 0; j < i + window_size; j++) {
				if (j < 0 || j >= X.rowSize()) {
					for (int k = 0; k < X.colSize(); k++) {
						y.set(u++, 0);
					}
				} else {
					DenseVector x = X.row(j);
					for (int k = 0; k < x.size(); k++) {
						y.set(u++, x.value(k));
					}
				}
			}
			y.summation();
		}

		return Y;
	}

	public int getEmbeddingSize() {
		return embedding_size;
	}

	@Override
	public int getInputSize() {
		return 0;
	}

	@Override
	public int getOutputSize() {
		return output_size;
	}

	@Override
	public DenseMatrix getW() {
		return null;
	}

	public int getWindowSize() {
		return window_size;
	}

}
