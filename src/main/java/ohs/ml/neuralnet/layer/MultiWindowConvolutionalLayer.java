package ohs.ml.neuralnet.layer;

import java.util.List;

import org.apache.thrift.server.TExtensibleServlet;

import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
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

	private List<ConvolutionalLayer> cls;

	private int emb_size;

	private int num_filters;

	private DenseMatrix tmp_dC = new DenseMatrix(0);

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private IntegerArray windowSizes;

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
		this.emb_size = emb_size;
		this.windowSizes = new IntegerArray(window_sizes);
		this.num_filters = num_filters;

		cls = Generics.newArrayList(window_sizes.length);

		for (int window_size : window_sizes) {
			cls.add(new ConvolutionalLayer(emb_size, window_size, num_filters));
		}
	}

	public MultiWindowConvolutionalLayer(List<ConvolutionalLayer> cls) {
		this.cls = cls;
		windowSizes = new IntegerArray(cls.size());

		for (int i = 0; i < cls.size(); i++) {
			windowSizes.add(cls.get(i).getWindowSize());
		}

		this.num_filters = cls.get(0).getNumFilters();
		this.emb_size = cls.get(0).getEmbeddingSize();
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		DenseTensor X = this.X;
		DenseTensor dX = new DenseTensor();

		dX.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_dC, dY.sizeOfInnerVectors(), emb_size);
		VectorUtils.enlarge(tmp_dX, X.sizeOfInnerVectors(), emb_size);

		List<DenseTensor> dT = Generics.newArrayList(windowSizes.size());

		for (int i = 0; i < windowSizes.size(); i++) {
			DenseTensor dC = new DenseTensor();
			dC.ensureCapacity(X.size());
			dT.add(dC);
		}

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix dYm = dY.get(i);

			int start = 0;
			for (int j = 0; j < windowSizes.size(); j++) {
				DenseMatrix dCm = dYm.subMatrix(start, Xm.rowSize());
				DenseTensor dC = dT.get(j);
				dC.add(dCm);

				start += Xm.rowSize();
			}
		}

		for (int i = 0, start = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix dXm = tmp_dX.subMatrix(start, Xm.rowSize());
			dXm.setAll(0);

			start += Xm.rowSize();

			dX.add(dXm);
		}

		for (int i = 0; i < windowSizes.size(); i++) {
			DenseTensor dC = dT.get(i);
			DenseTensor dZ = cls.get(i).backward(dC);

			for (int j = 0; j < dZ.size(); j++) {
				DenseMatrix dZm = dZ.row(j);
				DenseMatrix dXm = dX.row(j);
				VectorMath.add(dZm, dXm);
			}
		}

		return dX;

	}

	@Override
	public Layer copy() {
		List<ConvolutionalLayer> ret = Generics.newArrayList(cls.size());

		for (int i = 0; i < cls.size(); i++) {
			ret.add(cls.get(i).copy());
		}

		return new MultiWindowConvolutionalLayer(ret);
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

		List<DenseTensor> T = Generics.newArrayList(windowSizes.size());

		for (int i = 0; i < windowSizes.size(); i++) {
			T.add(cls.get(i).forward(X));
		}

		DenseTensor Y = new DenseTensor();
		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors() * windowSizes.size(), num_filters);
		int start = 0;

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			int num_feat_maps = Xm.rowSize() * windowSizes.size();

			DenseMatrix Ym = tmp_Y.subMatrix(start, num_feat_maps);
			Ym.setAll(0);

			start += num_feat_maps;

			int j = 0;

			for (DenseTensor C : T) {
				DenseMatrix Cm = C.get(i);
				for (DenseVector c : Cm) {
					DenseVector y = Ym.row(j++);
					VectorMath.add(c, y);
				}
			}

			Y.add(Ym);
		}

		this.Y = Y;
		return Y;
	}

	@Override
	public DenseTensor getB() {
		DenseTensor ret = new DenseTensor();
		for (Layer l : cls) {
			ret.addAll(l.getB());
		}
		return ret;
	}

	@Override
	public DenseTensor getDB() {
		DenseTensor ret = new DenseTensor();
		for (Layer l : cls) {
			ret.addAll(l.getDB());
		}
		return ret;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor ret = new DenseTensor();
		for (Layer l : cls) {
			for (DenseMatrix dW : l.getDW()) {
				ret.add(dW);
			}
		}
		return ret;
	}

	@Override
	public DenseTensor getW() {
		DenseTensor ret = new DenseTensor();
		for (Layer l : cls) {
			for (DenseMatrix W : l.getW()) {
				ret.add(W);
			}
		}
		return ret;
	}

	public IntegerArray getWindowSizes() {
		return windowSizes;
	}

	@Override
	public void init() {
		for (Layer l : cls) {
			l.init();
		}
	}

	@Override
	public void prepare() {
		for (Layer l : cls) {
			l.prepare();
		}
	}

}
