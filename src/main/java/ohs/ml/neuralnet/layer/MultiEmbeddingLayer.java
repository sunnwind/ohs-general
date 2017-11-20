package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.io.FileUtils;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.utils.Generics;

public class MultiEmbeddingLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private DenseTensor X;

	private DenseTensor W;

	private DenseTensor dW;

	private boolean forward_feature_indexes = true;

	private boolean[] learn_embeddings;

	private int new_emb_size;

	public MultiEmbeddingLayer(DenseTensor W, boolean[] learn_embeddings) {
		this.W = W;

		for (DenseMatrix E : W) {
			new_emb_size += E.colSize();
		}
		this.learn_embeddings = learn_embeddings;
	}

	public MultiEmbeddingLayer(int[][] sizes, boolean[] learn_embeddings) {
		W = new DenseTensor();
		W.ensureCapacity(sizes.length);

		for (int i = 0; i < sizes.length; i++) {
			int feat_size = sizes[i][0];
			int emb_size = sizes[i][1];
			new_emb_size += emb_size;
			W.add(new DenseMatrix(feat_size, emb_size));
		}
		this.learn_embeddings = learn_embeddings;
	}

	@Override
	public Object backward(Object I) {
		DenseTensor X = (DenseTensor) this.X;
		DenseTensor dY = (DenseTensor) I;

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix dYm = dY.get(i);

			for (int j = 0; j < Xm.rowSize(); j++) {
				DenseVector xm = Xm.row(j);
				DenseVector dym = dYm.row(j);
				int new_idx = 0;

				for (int feat_idx = 0; feat_idx < xm.size(); feat_idx++) {
					if (learn_embeddings[feat_idx]) {
						int val_idx = (int) xm.value(feat_idx);
						DenseVector dw = dW.get(feat_idx).row(val_idx);

						for (int l = 0; l < dw.size(); l++) {
							dw.add(l, dym.value(new_idx++));
						}
					}
				}
			}
		}

		return null;

	}

	public MultiEmbeddingLayer copy() {
		MultiEmbeddingLayer l = new MultiEmbeddingLayer(W, learn_embeddings);
		l.setOutputWordIndexes(forward_feature_indexes);
		return l;
	}

	@Override
	public Object forward(Object I) {
		this.X = (DenseTensor) I;
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();

		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), new_emb_size);
		int start = 0;

		for (DenseMatrix Xm : X) {
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			Ym.setAll(0);

			start += Xm.rowSize();

			for (int i = 0; i < Xm.rowSize(); i++) {
				DenseVector xm = Xm.row(i);
				DenseVector ym = Ym.row(i);
				int sum = 0;
				int new_idx = 0;

				for (int feat_idx = 0; feat_idx < xm.size(); feat_idx++) {
					int val_idx = (int) xm.value(feat_idx);
					DenseVector e = W.get(feat_idx).row(val_idx);

					for (int k = 0; k < e.size(); k++) {
						ym.add(new_idx++, e.value(k));
					}
				}
			}

			Y.add(Ym);
		}

		if (forward_feature_indexes) {
			return Generics.newPair(X, Y);
		} else {
			return Y;
		}
	}

	@Override
	public DenseTensor getDW() {
		return dW;
	}

	@Override
	public int getOutputSize() {
		return new_emb_size;
	}

	@Override
	public DenseTensor getW() {
		return W;
	}

	@Override
	public void initWeights() {
		for (DenseMatrix E : W) {
			ParameterInitializer.init2(E);
		}
	}

	@Override
	public void prepareTraining() {
		dW = new DenseTensor();
		dW.ensureCapacity(W.size());

		for (DenseMatrix Wm : W) {
			dW.add(Wm.copy(true));
		}
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		learn_embeddings = FileUtils.readBooleans(ois);
		W.readObject(ois);
	}

	public void setOutputWordIndexes(boolean output_word_indexes) {
		this.forward_feature_indexes = output_word_indexes;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		FileUtils.write(oos, learn_embeddings);
		W.writeObject(oos);
	}

}
