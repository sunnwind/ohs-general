package ohs.ml.neuralnet.layer;

import java.util.List;

import org.apache.poi.hssf.util.HSSFColor.DARK_TEAL;

import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.utils.Generics;

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
public class MultiWindowConvolutionalLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -997396529874765085L;

	/**
	 * filter biases
	 */
	private DenseVector b;

	private List<ConvolutionalLayer> cns;

	private DenseVector db;

	private DenseMatrix dW;

	private int emb_size;

	/**
	 * embedding size x window size
	 */
	private int filter_size;

	private int num_filters;

	private int pad_size;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_dZ = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	public int total_filters;

	/**
	 * filters x filter size (embedding size x window size)
	 */
	private DenseMatrix W;

	private int[] window_sizes;

	/**
	 * data x words x embeddings
	 */
	private DenseTensor X;

	/**
	 * data x feature maps x filter size
	 */
	private DenseTensor Y;

	/**
	 * data x feature maps x filter size
	 */
	private DenseTensor Z;

	public MultiWindowConvolutionalLayer(int emb_size, int[] window_sizes, int num_filters) {

		cns = Generics.newArrayList(window_sizes.length);

		for (int window_size : window_sizes) {
			cns.add(new ConvolutionalLayer(emb_size, window_size, num_filters));
		}

		this.window_sizes = window_sizes;
		this.num_filters = num_filters;
		this.emb_size = emb_size;
		this.total_filters = window_sizes.length * num_filters;
	}

	public MultiWindowConvolutionalLayer(List<ConvolutionalLayer> cns) {
		this.cns = cns;
		window_sizes = new int[cns.size()];

		for (int i = 0; i < cns.size(); i++) {
			window_sizes[i] = cns.get(i).getWindowSize();
		}

		emb_size = cns.get(0).getEmbeddingSize();

		this.total_filters = window_sizes.length * cns.get(0).getInputSize();
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		int data_size = dY.size();

		int num_vecs = dY.sizeOfInnerVectors() * window_sizes.length;

		{

			if (tmp_dZ.rowSize() < num_vecs) {
				tmp_dZ = new DenseMatrix(num_vecs, num_filters);
			}
		}

		int num_windows = window_sizes.length;

		List<DenseTensor> L1 = Generics.newArrayList(num_windows);

		for (int i = 0, start = 0; i < num_windows; i++) {
			DenseTensor T = new DenseTensor();
			T.ensureCapacity(data_size);

			for (int j = 0; j < data_size; j++) {
				int len = dY.row(j).rowSize();
				T.add(tmp_dZ.rows(start, len));
				start += len;
			}
			L1.add(T);
		}

		for (int i = 0, start = 0; i < num_windows; i++) {
			DenseTensor Z = L1.get(i);
			for (int j = 0; j < dY.rowSize(); j++) {
				DenseMatrix dYm = dY.row(j);
				DenseMatrix Zm = Z.row(j);

				for (int k = 0; k < dYm.rowSize(); k++) {
					DenseVector dy = dYm.row(k);
					DenseVector z = Zm.row(k);

					ArrayUtils.copy(dy.values(), start, z.values(), 0, num_filters);
				}
			}
			start += num_filters;
		}

		List<DenseTensor> L2 = Generics.newArrayList(num_windows);

		for (int i = 0; i < num_windows; i++) {
			DenseTensor dZ = cns.get(i).backward(L1.get(i));
			L2.add(dZ);
		}

		if (tmp_dX.rowSize() < num_vecs) {
			tmp_dX = new DenseMatrix(num_vecs, emb_size);
		}

		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(data_size);

		for (int i = 0, start = 0; i < data_size; i++) {
			int len = dY.row(i).rowSize();
			DenseMatrix dXm = tmp_dX.rows(start, len);
			dXm.setAll(0);

			dX.add(dXm);
			start += len;
		}

		for (int i = 0; i < num_windows; i++) {
			DenseTensor dZ = L2.get(i);

			for (int j = 0; j < data_size; j++) {
				DenseMatrix dZm = dZ.row(j);
				DenseMatrix dXm = dX.row(j);

				for (int k = 0; k < dZm.rowSize(); k++) {
					DenseVector dz = dZm.row(k);
					DenseVector dx = dXm.row(k);
					VectorMath.add(dz, dx);
				}
			}
		}

		return dX;

	}

	@Override
	public Layer copy() {
		return new MultiWindowConvolutionalLayer(cns);
	}

	/*
	 * I = data size x feature maps x filters
	 */
	@Override
	public DenseTensor forward(Object I) {
		List<DenseTensor> L = Generics.newArrayList(cns.size());

		for (ConvolutionalLayer cn : cns) {
			L.add(cn.forward(I));
		}

		DenseTensor X = (DenseTensor) I;
		this.X = X;

		{
			int len = X.sizeOfInnerVectors();
			if (tmp_Y.rowSize() < len) {
				tmp_Y = new DenseMatrix(len, num_filters * window_sizes.length);
			}
		}

		Y = new DenseTensor();
		Y.ensureCapacity(X.rowSize());

		for (int i = 0, start = 0; i < X.rowSize(); i++) {
			int len = X.row(i).rowSize();
			DenseMatrix Ym = tmp_Y.rows(start, len);
			Y.add(Ym);
			start += len;
		}

		for (int i = 0, start = 0; i < L.size(); i++) {
			DenseTensor Z = L.get(i);
			for (int j = 0; j < Z.rowSize(); j++) {
				DenseMatrix Zm = Z.row(j);
				DenseMatrix Ym = Y.row(j);

				for (int k = 0; k < Zm.rowSize(); k++) {
					DenseVector z = Zm.row(k);
					DenseVector y = Ym.row(k);
					ArrayUtils.copy(z.values(), 0, y.values(), start, z.size());
				}
			}
			start += num_filters;
		}
		return Y;
	}

	@Override
	public DenseMatrix getB() {
		DenseMatrix ret = new DenseMatrix();
		for (Layer l : cns) {
			ret.addAll(l.getB());
		}
		return ret;
	}

	@Override
	public DenseMatrix getDB() {
		DenseMatrix ret = new DenseMatrix();
		for (Layer l : cns) {
			ret.addAll(l.getDB());
		}
		return ret;
	}

	@Override
	public DenseMatrix getDW() {
		DenseMatrix ret = new DenseMatrix();
		for (Layer l : cns) {
			ret.addAll(l.getDW());
		}
		return ret;
	}

	public int getEmbeddingSize() {
		return emb_size;
	}

	@Override
	public int getInputSize() {
		return W.rowSize();
	}

	@Override
	public int getOutputSize() {
		return W.colSize();
	}

	public int getTotalFilters() {
		return total_filters;
	}

	@Override
	public DenseMatrix getW() {
		DenseMatrix ret = new DenseMatrix();
		for (Layer l : cns) {
			ret.addAll(l.getW());
		}
		return ret;
	}

	public int[] getWindowSizes() {
		return window_sizes;
	}

	@Override
	public void init() {
		for (Layer l : cns) {
			l.init();
		}
	}

	@Override
	public void prepare() {
		for (Layer l : cns) {
			l.prepare();
		}
	}

}
