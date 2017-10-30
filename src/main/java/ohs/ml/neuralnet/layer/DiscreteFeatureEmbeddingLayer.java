package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.apache.poi.hssf.record.common.FeatFormulaErr2;

import ohs.math.ArrayUtils;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.types.number.IntegerTensor;

public class DiscreteFeatureEmbeddingLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private DenseMatrix dW;

	private int extra_emb_size;

	private DenseTensor F;

	private boolean is_learning = true;

	private int new_emb_size;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	/**
	 * words x embedding size
	 */
	private DenseMatrix W;

	private int word_emb_size;

	private DenseTensor X;

	private DenseTensor Y;

	public DiscreteFeatureEmbeddingLayer() {

	}

	public DiscreteFeatureEmbeddingLayer(DenseMatrix W, int word_emb_size, boolean is_learning) {
		this.W = W;
		this.word_emb_size = word_emb_size;
		this.is_learning = is_learning;

		extra_emb_size = W.rowSize() * W.colSize();
		new_emb_size = word_emb_size + extra_emb_size;
	}

	public DiscreteFeatureEmbeddingLayer(int feat_size, int feat_emb_size, int word_emb_size, boolean learn_embedding) {
		this(new DenseMatrix(feat_size, feat_emb_size), word_emb_size, learn_embedding);
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		DenseTensor dX = new DenseTensor();
		DenseTensor X = this.X;

		dX.ensureCapacity(dY.size());

		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), word_emb_size);

		int start = 0;
		int feat_size = W.rowSize();
		int feat_emb_size = W.colSize();

		for (int u = 0; u < dY.size(); u++) {
			DenseMatrix dYm = dY.get(u);
			DenseMatrix Xm = X.get(u);
			DenseMatrix dXm = tmp_dX.subMatrix(start, dYm.rowSize());
			dXm.setAll(0);

			start += dYm.rowSize();

			for (int i = 0; i < dYm.rowSize(); i++) {
				DenseVector dy = dYm.row(i);
				DenseVector dx = dXm.row(i);

				for (int j = 0; j < word_emb_size; j++) {
					dx.add(j, dy.value(j));
				}

				int pos = word_emb_size;

				if (is_learning) {
					for (int j = 0; j < feat_size; j++) {
						DenseVector dw = dW.row(j);
						for (int k = 0; k < feat_emb_size; k++) {
							dw.add(k, dy.value(pos++));
						}
					}
				}
			}
			dX.add(dXm);
		}

		return dX;
	}

	public Layer copy() {
		return new DiscreteFeatureEmbeddingLayer(W, word_emb_size, is_learning);
	}

	@Override
	public Object forward(Object I) {

		Pair<DenseTensor, DenseTensor> p = (Pair<DenseTensor, DenseTensor>) I;
		F = p.getFirst();
		X = p.getSecond();

		DenseTensor X = this.X;
		DenseTensor F = this.F;
		DenseTensor Y = new DenseTensor();
		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), new_emb_size);

		int start = 0;
		int feat_emb_size = W.colSize();

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.row(i);
			DenseMatrix Fm = F.row(i);
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			Ym.setAll(0);

			start += Xm.rowSize();

			for (int j = 0; j < Xm.rowSize(); j++) {
				DenseVector f = Fm.row(j);
				DenseVector x = Xm.row(j);
				DenseVector y = Ym.row(j);

				for (int k = 0, pos = 0; k < f.size(); k++) {
					if (k == 0) {
						ArrayUtils.copy(x.values(), 0, y.values(), pos, word_emb_size);
						pos += word_emb_size;
					} else {
						double feat_flag = f.value(k);

						if (feat_flag == 1d) {
							int feat_idx = k - 1;
							DenseVector w = W.row(feat_idx);
							ArrayUtils.copy(w.values(), 0, y.values(), pos, feat_emb_size);
						}
						pos += feat_emb_size;
					}
				}
			}
			Y.add(Ym);
		}

		this.Y = Y;

		return Y;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor ret = new DenseTensor();
		ret.add(dW);
		return ret;
	}

	@Override
	public int getOutputSize() {
		return new_emb_size;
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
		return is_learning;
	}

	@Override
	public void prepare() {
		dW = W.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		W = new DenseMatrix(ois);
		is_learning = ois.readBoolean();
		word_emb_size = ois.readInt();

		extra_emb_size = W.rowSize() * W.colSize();
		new_emb_size = word_emb_size + extra_emb_size;
	}

	public void setLearnEmbedding(boolean is_learning) {
		this.is_learning = is_learning;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		W.writeObject(oos);
		oos.writeBoolean(is_learning);
		oos.writeInt(word_emb_size);
	}

}
