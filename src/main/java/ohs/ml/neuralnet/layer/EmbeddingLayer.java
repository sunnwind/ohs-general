package ohs.ml.neuralnet.layer;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.types.number.IntegerArray;

public class EmbeddingLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private DenseMatrix W;

	private DenseMatrix dW;

	private DenseMatrix tmp_Y;

	private boolean learn_embedding = true;

	private IntegerArray X;

	private DenseMatrix Y;

	public EmbeddingLayer(DenseMatrix W, boolean learn_embedding) {
		this.W = W;
		this.learn_embedding = learn_embedding;
	}

	public EmbeddingLayer(int vocab_size, int embedding_size, boolean learn_embedding) {
		this(new DenseMatrix(vocab_size, embedding_size), learn_embedding);
	}

	@Override
	public Object backward(Object I) {
		if (I != null && learn_embedding) {
			DenseMatrix dY = (DenseMatrix) I;
			for (int i = 0; i < X.size(); i++) {
				int w = X.get(i);
				VectorMath.add(dY.row(i), dW.row(w));
			}
		}
		return null;
	}

	@Override
	public DenseMatrix forward(Object I) {
		X = (IntegerArray) I;

		int data_size = X.size();
		int embedding_size = W.colSize();

		if (tmp_Y == null || tmp_Y.rowSize() < data_size) {
			tmp_Y = new DenseMatrix(data_size, embedding_size);
		}

		Y = tmp_Y.rowsAsMatrix(data_size);

		for (int i = 0; i < data_size; i++) {
			int w = X.get(i);
			VectorUtils.copy(W.row(w), Y.row(i));
		}
		return Y;
	}

	@Override
	public DenseMatrix getDW() {
		return dW;
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
		ParameterInitializer.init2(W);
	}

	public boolean isLearnEmbedding() {
		return learn_embedding;
	}

	@Override
	public void prepare() {
		dW = W.copy(true);
	}

	public void setLearnEmbedding(boolean learn_embedding) {
		this.learn_embedding = learn_embedding;
	}

}
