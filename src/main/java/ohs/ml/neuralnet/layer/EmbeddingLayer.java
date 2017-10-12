package ohs.ml.neuralnet.layer;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;

public class EmbeddingLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private DenseMatrix dW;

	private boolean learn_embedding = true;

	/**
	 * words x embedding size
	 */
	private DenseMatrix W;

	private Object X;

	private Object Y;

	public EmbeddingLayer(DenseMatrix W, boolean learn_embedding) {
		this.W = W;
		this.learn_embedding = learn_embedding;
	}

	public EmbeddingLayer(int vocab_size, int emb_size, boolean learn_embedding) {
		this(new DenseMatrix(vocab_size, emb_size), learn_embedding);
	}

	@Override
	public Object backward(Object I) {
		if (I != null && learn_embedding) {
			if (I instanceof DenseMatrix) {
				IntegerArray X = (IntegerArray) this.X;
				DenseMatrix dY = (DenseMatrix) I;
				for (int i = 0; i < X.size(); i++) {
					int w = X.get(i);
					VectorMath.add(dY.row(i), dW.row(w));
				}
			} else if (I instanceof DenseTensor) {
				IntegerMatrix X = (IntegerMatrix) this.X;
				DenseTensor dY = (DenseTensor) I;
				for (int i = 0; i < X.size(); i++) {
					IntegerArray Xm = X.get(i);
					DenseMatrix dYm = dY.get(i);

					for (int j = 0; j < Xm.size(); j++) {
						int w = Xm.get(j);
						try {
							VectorMath.add(dYm.row(j), dW.row(w));
						} catch (Exception e) {
							e.printStackTrace();
						}
					}
				}
			}
		}
		return null;
	}

	public Layer copy() {
		return new EmbeddingLayer(W, learn_embedding);
	}

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	@Override
	public Object forward(Object I) {
		this.X = I;

		if (I instanceof IntegerArray) {
			IntegerArray X = (IntegerArray) I;
			int data_size = X.size();
			VectorUtils.enlarge(tmp_Y, data_size, W.colSize());
			DenseMatrix Wm = W.rows(X.values());
			DenseMatrix Y = tmp_Y.rows(data_size);
			VectorUtils.copy(Wm, Y);
			this.Y = Y;
		} else if (I instanceof IntegerMatrix) {
			IntegerMatrix X = (IntegerMatrix) I;
			DenseTensor Y = new DenseTensor();
			Y.ensureCapacity(X.size());

			VectorUtils.enlarge(tmp_Y, X.sizeOfEntries(), W.colSize());
			int start = 0;
			for (IntegerArray Xm : X) {
				DenseMatrix Wm = W.rows(Xm.values());
				DenseMatrix Ym = tmp_Y.rows(start, Xm.size());
				VectorUtils.copy(Wm, Ym);
				Y.add(Ym);
				start += Xm.size();
			}
			this.Y = Y;
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
