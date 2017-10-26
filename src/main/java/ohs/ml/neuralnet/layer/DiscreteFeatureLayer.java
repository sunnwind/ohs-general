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

public class DiscreteFeatureLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private DenseMatrix dW;

	private int extra_emb_size;

	private IntegerMatrix F;

	private boolean learn_embedding = true;

	private int new_emb_size;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	/**
	 * words x embedding size
	 */
	private DenseMatrix W;

	private int word_emb_size;

	private DenseMatrix X;

	private DenseMatrix Y;

	public DiscreteFeatureLayer() {

	}

	public DiscreteFeatureLayer(DenseMatrix W, int word_emb_size, boolean learn_embedding) {
		this.W = W;
		this.word_emb_size = word_emb_size;
		this.learn_embedding = learn_embedding;

		extra_emb_size = W.rowSize() * W.colSize();
		new_emb_size = word_emb_size + extra_emb_size;
	}

	public DiscreteFeatureLayer(int feat_size, int feat_emb_size, int word_emb_size, boolean learn_embedding) {
		this(new DenseMatrix(feat_size, feat_emb_size), word_emb_size, learn_embedding);
	}

	@Override
	public Object backward(Object I) {
		DenseMatrix dY = (DenseMatrix) I;

		VectorUtils.enlarge(tmp_dX, dY.rowSize(), word_emb_size);

		DenseMatrix dX = tmp_dX.rows(dY.rowSize());
		dX.setAll(0);

		int feat_size = W.rowSize();
		int feat_emb_size = W.colSize();

		for (int i = 0; i < dY.rowSize(); i++) {
			DenseVector dy = dY.row(i);
			DenseVector dx = dX.row(i);

			for (int j = 0; j < word_emb_size; j++) {
				dx.add(j, dy.value(j));
			}

			int pos = word_emb_size;

			for (int j = 0; j < feat_size; j++) {
				DenseVector dw = dW.row(j);
				for (int k = 0; k < feat_emb_size; k++) {
					dw.add(k, dy.value(pos++));
				}
			}

		}

		return dX;
	}

	public Layer copy() {
		return new DiscreteFeatureLayer(W, word_emb_size, learn_embedding);
	}

	@Override
	public Object forward(Object I) {

		Pair<IntegerTensor, DenseMatrix> p = (Pair<IntegerTensor, DenseMatrix>) I;
		F = p.getFirst().get(0);
		X = p.getSecond();

		int data_size = X.rowSize();

		VectorUtils.enlarge(tmp_Y, data_size, new_emb_size);

		DenseMatrix Y = tmp_Y.rows(data_size);
		Y.setAll(0);

		int feat_emb_size = W.colSize();

		for (int i = 0; i < data_size; i++) {
			DenseVector x = X.row(i);
			DenseVector y = Y.row(i);
			IntegerArray f = F.get(i);

			for (int j = 0, pos = 0; j < f.size(); j++) {
				if (j == 0) {
					ArrayUtils.copy(x.values(), 0, y.values(), pos, word_emb_size);
					pos += word_emb_size;
				} else {
					int feat_flag = f.get(j);
					if (feat_flag == 1) {
						int feat_idx = j - 1;
						DenseVector w = W.row(feat_idx);
						ArrayUtils.copy(w.values(), 0, y.values(), pos, feat_emb_size);
					}
					pos += feat_emb_size;
				}
			}
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
		return learn_embedding;
	}

	@Override
	public void prepare() {
		dW = W.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		W = new DenseMatrix(ois);
		learn_embedding = ois.readBoolean();
		word_emb_size = ois.readInt();

		extra_emb_size = W.rowSize() * W.colSize();
		new_emb_size = word_emb_size + extra_emb_size;
	}

	public void setLearnEmbedding(boolean learn_embedding) {
		this.learn_embedding = learn_embedding;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		W.writeObject(oos);
		oos.writeBoolean(learn_embedding);
		oos.writeInt(word_emb_size);
	}

}
