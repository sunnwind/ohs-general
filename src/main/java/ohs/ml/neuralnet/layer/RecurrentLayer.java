package ohs.ml.neuralnet.layer;

import ohs.matrix.DenseMatrix;

public abstract class RecurrentLayer extends Layer {

	public static enum Type {
		LSTM, RNN
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 5913734004543729479L;

	protected DenseMatrix dWhh;

	protected DenseMatrix dWxh;

	/**
	 * hidden to hidden
	 */
	protected DenseMatrix Whh;

	/**
	 * input to hidden
	 */
	protected DenseMatrix Wxh;

	public RecurrentLayer() {

	}

	public DenseMatrix getDWhh() {
		return dWhh;
	}

	public DenseMatrix getDWxh() {
		return dWxh;
	}

	public DenseMatrix getWhh() {
		return Whh;
	}

	public DenseMatrix getWxh() {
		return Wxh;
	}

	public abstract void resetH0();

}
