package ohs.ml.neuralnet.layer;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;

public abstract class RecurrentLayer extends Layer {

	public static enum Type {
		LSTM, RNN
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 5913734004543729479L;

	protected DenseMatrix Wxh;

	protected DenseMatrix Whh;

	protected DenseVector b;

	protected DenseMatrix dWxh;

	protected DenseMatrix dWhh;

	protected DenseVector db;

	protected int shift_size = 1;

	protected int window_size = 1;

	/**
	 * Semeniuta, S., Severyn, A., & Barth, E. (2016). Recurrent Dropout without
	 * Memory Loss. Retrieved from http://arxiv.org/abs/1603.05118
	 * 
	 * @param X
	 * @param p
	 */
	public static void dropout(DenseMatrix X, double p) {
		DenseMatrix M = X.copy(true);
		VectorMath.random(0, 1, M);
		VectorMath.mask(M, p);
		M.multiply(1f / p); // inverted drop-out
		VectorMath.multiply(X, M, X);
	}

	public RecurrentLayer() {

	}

	public DenseMatrix getDWhh() {
		return dWhh;
	}

	public DenseMatrix getDWxh() {
		return dWxh;
	}

	public int getShiftSize() {
		return shift_size;
	}

	public DenseMatrix getWhh() {
		return Whh;
	}

	public int getWindowSize() {
		return window_size;
	}

	public DenseMatrix getWxh() {
		return Wxh;
	}

	public abstract void resetH0();

	public void setShiftSize(int shift_size) {
		this.shift_size = shift_size;
	}

	public void setWindowSize(int window_size) {
		this.window_size = window_size;
	}

}
