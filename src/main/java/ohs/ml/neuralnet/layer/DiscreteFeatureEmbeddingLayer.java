package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.apache.poi.hssf.record.common.FeatFormulaErr2;

import ohs.math.ArrayUtils;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.NERFeatureExtractor;
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

	private int new_emb_size;

	private int prev_emb_size;

	private int extra_emb_size;

	private int feat_size;

	private boolean learn_embs = true;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	/**
	 * words x embedding size
	 */
	private DenseMatrix W;

	private DenseMatrix dW;

	private DenseTensor F;

	private DenseTensor X;

	private DenseTensor Y;

	public DiscreteFeatureEmbeddingLayer() {

	}

	public DiscreteFeatureEmbeddingLayer(DenseMatrix W, int feat_size, int prev_emb_size, boolean learn_embs) {
		this.W = W;
		this.prev_emb_size = prev_emb_size;
		this.learn_embs = learn_embs;
		this.feat_size = feat_size;

		extra_emb_size = feat_size * W.colSize();
		new_emb_size = prev_emb_size + extra_emb_size;
	}

	public DiscreteFeatureEmbeddingLayer(int feat_val_size, int feat_size, int feat_emb_size, int prev_emb_size,
			boolean learn_embedding) {
		this(new DenseMatrix(feat_val_size, feat_emb_size), feat_size, prev_emb_size, learn_embedding);
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		DenseTensor dX = new DenseTensor();
		DenseTensor X = this.X;
		DenseTensor F = this.F;

		dX.ensureCapacity(dY.size());

		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), prev_emb_size);

		int start = 0;
		int feat_size = W.rowSize();
		int feat_emb_size = W.colSize();

		for (int u = 0; u < dY.size(); u++) {
			DenseMatrix dYm = dY.get(u);
			DenseMatrix Xm = X.get(u);
			DenseMatrix Fm = F.get(u);
			DenseMatrix dXm = tmp_dX.subMatrix(start, dYm.rowSize());
			dXm.setAll(0);

			start += dYm.rowSize();

			for (int i = 0; i < dYm.rowSize(); i++) {
				DenseVector dy = dYm.row(i);
				DenseVector dx = dXm.row(i);
				DenseVector fm = Fm.row(i);

				for (int j = 0; j < prev_emb_size; j++) {
					dx.add(j, dy.value(j));
				}

				int pos = prev_emb_size;

				if (learn_embs) {
					for (int j = 1; j < fm.size(); j++) {
						int feat_idx = (int) fm.value(j);
						if (feat_idx >= 0) {
							DenseVector dw = dW.row(feat_idx);
							int _pos = pos;
							for (int k = 0; k < feat_emb_size; k++) {
								dw.add(k, dy.value(_pos++));
							}
						}
						pos += feat_emb_size;
					}
				}
			}
			dX.add(dXm);
		}

		return dX;
	}

	public Layer copy() {
		return new DiscreteFeatureEmbeddingLayer(W, feat_size, prev_emb_size, learn_embs);
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
				DenseVector fm = Fm.row(j);
				DenseVector xm = Xm.row(j);
				DenseVector ym = Ym.row(j);

				for (int k = 0, pos = 0; k < fm.size(); k++) {
					if (k == 0) {
						ArrayUtils.copy(xm.values(), 0, ym.values(), pos, prev_emb_size);
						pos += prev_emb_size;
					} else {
						int feat_idx = (int) fm.value(k);
						if (feat_idx >= 0) {
							DenseVector w = W.row(feat_idx);
							ArrayUtils.copy(w.values(), 0, ym.values(), pos, feat_emb_size);
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
	public void initWeights() {
		ParameterInitializer.init2(W);
	}

	public boolean isLearnEmbedding() {
		return learn_embs;
	}

	@Override
	public void prepare() {
		dW = W.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		W = new DenseMatrix(ois);
		prev_emb_size = ois.readInt();
		learn_embs = ois.readBoolean();
		feat_size = ois.readInt();

		extra_emb_size = feat_size * W.colSize();
		new_emb_size = prev_emb_size + extra_emb_size;
	}

	public void setLearnEmbedding(boolean learn_embs) {
		this.learn_embs = learn_embs;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		W.writeObject(oos);
		oos.writeInt(prev_emb_size);
		oos.writeBoolean(learn_embs);
		oos.writeInt(feat_size);

	}

}
