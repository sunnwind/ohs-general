package ohs.ml.neuralnet.layer;

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

	protected DenseVector b;

	protected int bptt_size = 5;

	protected DenseVector db;

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

	public int getBpttSize() {
		return bptt_size;
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

	public void setBpttSize(int bptt_size) {
		this.bptt_size = bptt_size;
	}

}
