package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.com.TaskType;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.types.number.IntegerTensor;
import ohs.utils.Generics;

public class EmbeddingLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private DenseMatrix dW;

	private boolean learn_embedding = true;

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private TaskType tt;

	/**
	 * words x embedding size
	 */
	private DenseMatrix W;

	private Object X;

	private Object Y;

	public EmbeddingLayer() {

	}

	public EmbeddingLayer(DenseMatrix W, boolean learn_embedding, TaskType tt) {
		this.W = W;
		this.learn_embedding = learn_embedding;
		this.tt = tt;
	}

	public EmbeddingLayer(int vocab_size, int emb_size, boolean learn_embedding, TaskType tt) {
		this(new DenseMatrix(vocab_size, emb_size), learn_embedding, tt);
	}

	@Override
	public Object backward(Object I) {
		if (I != null && learn_embedding) {
			if (tt == TaskType.SEQ_LABELING) {
				IntegerTensor X = (IntegerTensor) this.X;
				DenseMatrix dY = (DenseMatrix) I;
				for (IntegerMatrix Xm : X) {
					for (int i = 0; i < Xm.size(); i++) {
						int w = Xm.get(i).get(0);
						VectorMath.add(dY.row(i), dW.row(w));
					}
				}
			}

			// else if (I instanceof DenseTensor) {
			// IntegerMatrix X = (IntegerMatrix) this.X;
			// DenseTensor dY = (DenseTensor) I;
			// for (int i = 0; i < X.size(); i++) {
			// IntegerArray Xm = X.get(i);
			// DenseMatrix dYm = dY.get(i);
			//
			// for (int j = 0; j < Xm.size(); j++) {
			// int w = Xm.get(j);
			// VectorMath.add(dYm.row(j), dW.row(w));
			// }
			// }
			// }
		}
		return null;
	}

	public Object backwardOld(Object I) {
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
						VectorMath.add(dYm.row(j), dW.row(w));
					}
				}
			}
		}
		return null;
	}

	public Layer copy() {
		return new EmbeddingLayer(W, learn_embedding, tt);
	}

	@Override
	public Object forward(Object I) {
		this.X = I;

		Object ret = null;

		if (tt == TaskType.CLASSIFICATION) {
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
			ret = Y;
		} else if (tt == TaskType.SEQ_CLASSIFICATION) {
			IntegerArray X = (IntegerArray) I;
			int data_size = X.size();
			VectorUtils.enlarge(tmp_Y, data_size, W.colSize());
			DenseMatrix Wm = W.rows(X.values());
			DenseMatrix Y = tmp_Y.rows(data_size);
			VectorUtils.copy(Wm, Y);
			this.Y = Y;
			ret = Y;
		} else if (tt == TaskType.SEQ_LABELING) {
			IntegerTensor X = (IntegerTensor) I;

			int seq_len = X.sizeOfIntegerArrays();

			VectorUtils.enlarge(tmp_Y, seq_len, W.colSize());

			DenseMatrix Y = tmp_Y.rows(seq_len);
			Y.setAll(0);

			IntegerArray ws = new IntegerArray(seq_len);

			for (IntegerMatrix Xm : X) {
				for (IntegerArray xm : Xm) {
					int w = xm.get(0);
					ws.add(w);
				}
			}

			DenseMatrix Wm = W.rows(ws.values());
			VectorUtils.copy(Wm, Y);

			this.Y = Y;

			ret = Generics.newPair(X, Y);
		}

		// if (I instanceof IntegerArray) {
		// IntegerArray X = (IntegerArray) I;
		// int data_size = X.size();
		// VectorUtils.enlarge(tmp_Y, data_size, W.colSize());
		// DenseMatrix Wm = W.rows(X.values());
		// DenseMatrix Y = tmp_Y.rows(data_size);
		// VectorUtils.copy(Wm, Y);
		// this.Y = Y;
		// ret = Y;
		// } else if (I instanceof IntegerMatrix) {
		// IntegerMatrix X = (IntegerMatrix) I;
		// DenseTensor Y = new DenseTensor();
		// Y.ensureCapacity(X.size());
		//
		// VectorUtils.enlarge(tmp_Y, X.sizeOfEntries(), W.colSize());
		// int start = 0;
		// for (IntegerArray Xm : X) {
		// DenseMatrix Wm = W.rows(Xm.values());
		// DenseMatrix Ym = tmp_Y.rows(start, Xm.size());
		// VectorUtils.copy(Wm, Ym);
		// Y.add(Ym);
		// start += Xm.size();
		// }
		// this.Y = Y;
		// ret = Y;
		// } else if (I instanceof IntegerTensor) {
		// IntegerTensor X = (IntegerTensor) I;
		// DenseTensor Y = new DenseTensor();
		// Y.ensureCapacity(X.size());
		//
		// VectorUtils.enlarge(tmp_Y, X.sizeOfIntegerArrays(), W.colSize());
		//
		// int start = 0;
		// for (IntegerMatrix Xm : X) {
		// IntegerArray ws = new IntegerArray(Xm.size());
		//
		// for (IntegerArray xm : Xm) {
		// int w = xm.get(0);
		// ws.add(w);
		// }
		//
		// DenseMatrix Wm = W.rows(ws.values());
		// DenseMatrix Ym = tmp_Y.rows(start, Xm.size());
		// VectorUtils.copy(Wm, Ym);
		// Y.add(Ym);
		// start += Xm.size();
		// }
		// this.Y = Y;
		//
		// ret = Generics.newPair(X, Y);
		// }

		return ret;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor ret = new DenseTensor();
		ret.add(dW);
		return ret;
	}

	@Override
	public int getOutputSize() {
		return W.colSize();
	}

	@Override
	public DenseTensor getW() {
		DenseTensor ret = new DenseTensor();
		ret.add(W);
		return ret;
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

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		learn_embedding = ois.readBoolean();
		W = new DenseMatrix(ois);
	}

	public void setLearnEmbedding(boolean learn_embedding) {
		this.learn_embedding = learn_embedding;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeBoolean(learn_embedding);
		W.writeObject(oos);
	}

}
