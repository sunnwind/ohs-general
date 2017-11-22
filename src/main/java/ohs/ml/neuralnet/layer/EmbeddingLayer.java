package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class EmbeddingLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private boolean learn_embedding = true;

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	/**
	 * words x embedding size
	 */
	private DenseMatrix W;

	private DenseMatrix dW;

	private DenseTensor X;

	private DenseTensor Y;

	private boolean output_feature_indexes = true;

	private int target_idx = 0;

	public EmbeddingLayer() {

	}

	public EmbeddingLayer(DenseMatrix W, boolean learn_embedding, int target_idx) {
		this.W = W;
		this.learn_embedding = learn_embedding;
		this.target_idx = target_idx;
	}

	public EmbeddingLayer(int vocab_size, int emb_size, boolean learn_embedding, int target_idx) {
		this(new DenseMatrix(vocab_size, emb_size), learn_embedding, target_idx);
	}

	@Override
	public Object backward(Object I) {
		if (I != null && learn_embedding) {
			DenseTensor X = (DenseTensor) this.X;
			DenseTensor dY = (DenseTensor) I;

			for (int i = 0; i < X.size(); i++) {
				DenseMatrix Xm = X.get(i);
				DenseMatrix dYm = dY.get(i);

				for (int j = 0; j < Xm.rowSize(); j++) {
					DenseVector xm = Xm.row(j);
					DenseVector dym = dYm.row(j);
					int w = (int) xm.value(target_idx);

					VectorMath.add(dym, dW.row(w));
				}
			}
		}
		return null;
	}

	public EmbeddingLayer copy() {
		EmbeddingLayer l = new EmbeddingLayer(W, learn_embedding, target_idx);
		l.setOutputWordIndexes(output_feature_indexes);
		return l;
	}

	@Override
	public Object forward(Object I) {
		this.X = (DenseTensor) I;
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();

		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), W.colSize());
		int start = 0;

		for (DenseMatrix Xm : X) {
			IntegerArray targets = new IntegerArray(Xm.rowSize());

			for (DenseVector xm : Xm) {
				int w = (int) xm.value(target_idx);
				targets.add(w);
			}

			DenseMatrix Wm = W.subMatrix(targets.values());
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			start += Xm.rowSize();

			VectorUtils.copy(Wm, Ym);
			Y.add(Ym);
		}

		this.Y = Y;

		if (output_feature_indexes) {
			return Generics.newPair(X, Y);
		} else {
			return Y;
		}
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

	public int getTargetIndex() {
		return target_idx;
	}

	@Override
	public DenseTensor getW() {
		DenseTensor ret = new DenseTensor();
		ret.add(W);
		return ret;
	}

	@Override
	public void initWeights() {
		ParameterInitializer.init2(W);
	}

	public boolean isLearnEmbedding() {
		return learn_embedding;
	}

	@Override
	public void prepareTraining() {
		dW = W.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		learn_embedding = ois.readBoolean();
		target_idx = ois.readInt();
		W = new DenseMatrix(ois);
	}

	public void setLearnEmbedding(boolean learn_embedding) {
		this.learn_embedding = learn_embedding;
	}

	public void setOutputWordIndexes(boolean output_word_indexes) {
		this.output_feature_indexes = output_word_indexes;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeBoolean(learn_embedding);
		oos.writeInt(target_idx);
		W.writeObject(oos);
	}

}
