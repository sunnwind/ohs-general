package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.types.generic.Pair;

/**
 * http://cthorey.github.io./backprop_conv/
 * 
 * http://www.slideshare.net/kuwajima/cnnbp
 * 
 * http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
 * 
 * https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/
 * 
 * @author ohs
 *
 */
public class ConvolutionalLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -997396529874765085L;

	private int emb_size;

	/**
	 * embedding size x window size
	 */
	private int filter_size;

	private int num_filters;

	private int window_size;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_dZ = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private DenseMatrix tmp_Z = new DenseMatrix(0);

	/**
	 * filter biases
	 */
	private DenseVector b;

	private DenseVector db;

	private DenseMatrix dW;

	/**
	 * filters x filter size (embedding size x window size)
	 */
	private DenseMatrix W;

	/**
	 * data x words x embeddings
	 */
	private DenseTensor X;

	/**
	 * data x feature maps x filters
	 */
	private DenseTensor Y;

	/**
	 * data x feature maps x filters
	 */
	private DenseTensor Z;

	/**
	 * @param W
	 *            feature maps x filters
	 * @param b
	 * 
	 * @param window_size
	 * @param emb_size
	 */
	public ConvolutionalLayer(DenseMatrix W, DenseVector b, int window_size, int emb_size) {
		this.W = W;
		this.b = b;

		this.window_size = window_size;
		this.emb_size = emb_size;

		filter_size = W.rowSize();
		num_filters = W.colSize();
	}

	public ConvolutionalLayer(int emb_size, int window_size, int num_filters) {
		this(new DenseMatrix(window_size * emb_size, num_filters), new DenseVector(num_filters), window_size, emb_size);
	}

	public ConvolutionalLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public DenseTensor backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		int data_size = dY.size();

		VectorUtils.enlarge(tmp_dZ, dY.sizeOfInnerVectors(), filter_size);
		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), emb_size);

		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(data_size);

		for (int i = 0, start = 0; i < data_size; i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix Zm = Z.get(i);
			DenseMatrix Ym = Y.get(i);
			DenseMatrix dYm = dY.get(i);

			int len = dYm.rowSize();

			DenseMatrix dZm = tmp_dZ.subMatrix(start, len);
			DenseMatrix dXm = tmp_dX.subMatrix(start, len);

			dZm.setAll(0);
			dXm.setAll(0);

			for (int j = 0; j < len; j++) {
				VectorMath.outerProduct(Zm.row(j), dYm.row(j), dW, true);
				VectorMath.add(dYm.row(j), db);
			}

			VectorMath.productRows(dYm, W, dZm, false);

			for (int j = 0; j < len; j++) {
				DenseVector dx = dXm.row(j);
				DenseVector dz = dZm.row(j);

				/*
				 * copy value from k-th at a filter to l-th at an embedding
				 */

				for (int k = 0; k < filter_size; k++) {
					int l = k % emb_size;
					dx.add(l, dz.value(k));
				}
			}

			dXm.multiply(1d / (window_size * len));

			dX.add(dXm);

			start += len;

		}

		return dX;
	}

	@Override
	public ConvolutionalLayer copy() {
		return new ConvolutionalLayer(W, b, window_size, emb_size);
	}

	/*
	 * I = data size x feature maps x filters
	 */
	@Override
	public DenseTensor forward(Object I) {
		DenseTensor X = null;

		if (I instanceof Pair) {
			Pair<DenseTensor, DenseTensor> p = (Pair<DenseTensor, DenseTensor>) I;
			X = p.getSecond();
		} else {
			X = (DenseTensor) I;
		}

		this.X = X;

		DenseTensor Y = new DenseTensor();
		DenseTensor Z = new DenseTensor();

		Y.ensureCapacity(X.size());
		Z.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), num_filters);
		VectorUtils.enlarge(tmp_Z, X.sizeOfInnerVectors(), filter_size);

		for (int i = 0, start = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			int sent_len = Xm.rowSize();

			/*
			 * sentence length = # of feature maps
			 */
			int num_feat_maps = sent_len;

			/*
			 * Zm = feature maps x filter size (concatenated embeddings)
			 */

			DenseMatrix Zm = null;
			Zm = tmp_Z.subMatrix(start, num_feat_maps);
			Zm.setAll(0);

			/*
			 * Xm (sentence length, embedding size) => Zm (feature maps, embedding size *
			 * window size)
			 */

			for (int j = 0; j < num_feat_maps; j++) {
				DenseVector z = Zm.row(j);
				for (int k = j, l = 0; k < j + window_size && k < num_feat_maps; k++) {
					DenseVector x = Xm.row(k);
					ArrayUtils.copy(x.values(), 0, z.values(), l, x.size());
					l += x.size();
				}
			}

			Z.add(Zm);

			/*
			 * Ym = feature maps X filters
			 */

			DenseMatrix Ym = tmp_Y.subMatrix(start, num_feat_maps);
			Ym.setAll(0);

			VectorMath.product(Zm, W, Ym, false);

			VectorMath.add(Ym, b, Ym);

			Y.add(Ym);

			start += num_feat_maps;
		}

		this.Y = Y;
		this.Z = Z;
		this.Z = Z;

		return Y;
	}

	@Override
	public DenseTensor getB() {
		DenseTensor ret = new DenseTensor();
		ret.add(b.toDenseMatrix());
		return ret;
	}

	@Override
	public DenseTensor getDB() {
		DenseTensor ret = new DenseTensor();
		ret.add(db.toDenseMatrix());
		return ret;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor ret = new DenseTensor();
		ret.add(dW);
		return ret;
	}

	public int getEmbeddingSize() {
		return emb_size;
	}

	@Override
	public int getInputSize() {
		return W.rowSize();
	}

	public int getNumFilters() {
		return num_filters;
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

	public int getWindowSize() {
		return window_size;
	}

	@Override
	public void initWeights() {
		ParameterInitializer.init2(W);
	}

	@Override
	public void prepareTraining() {
		dW = W.copy(true);
		db = b.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		W = new DenseMatrix(ois);
		b = new DenseVector(ois);

		filter_size = W.rowSize();
		num_filters = W.colSize();
		window_size = ois.readInt();
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		W.writeObject(oos);
		b.writeObject(oos);
		oos.writeInt(window_size);
	}

}
