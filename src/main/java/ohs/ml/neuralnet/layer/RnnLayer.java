package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.apache.commons.math.stat.descriptive.SynchronizedMultivariateSummaryStatistics;

import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.nonlinearity.Nonlinearity;
import ohs.types.number.IntegerArray;

/**
 *
 * 
 * Sutskever, I. (2013). Training Recurrent neural Networks. PhD thesis.
 * University of Toronto.
 * 
 * https://jramapuram.github.io/ramblings/rnn-backrpop/
 * 
 * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 * 
 * https://gist.github.com/karpathy/d4dee566867f8291f086
 * 
 * http://r2rt.com/styles-of-truncated-backpropagation.html
 * 
 * http://stats.stackexchange.com/questions/219914/rnns-when-to-apply-bptt-and-or-update-weights
 * 
 * http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
 * 
 * @author ohs
 */
public class RnnLayer extends RecurrentLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2848535801786544922L;

	private DenseMatrix tmp_dZ = new DenseMatrix(0);

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_H = new DenseMatrix(0);

	private DenseVector dh_next = new DenseVector(0);

	private DenseVector dh_raw = new DenseVector(0);

	private DenseVector dh_raw2 = new DenseVector(0);

	private DenseVector h0 = new DenseVector(0);

	private int hidden_size;

	private int input_size;

	private Nonlinearity non;

	private DenseTensor X;

	private DenseTensor H;

	private DenseTensor dX;

	private DenseTensor dH;

	public RnnLayer() {

	}

	public RnnLayer(DenseMatrix Wxh, DenseMatrix Whh, DenseVector bh, int bptt_size, Nonlinearity non) {
		super();
		this.Wxh = Wxh;
		this.Whh = Whh;
		this.b = bh;
		this.bptt_size = bptt_size;
		this.non = non;

		this.input_size = Wxh.rowSize();
		this.hidden_size = Wxh.colSize();
	}

	public RnnLayer(int input_size, int hidden_size, int bptt_size, Nonlinearity non) {
		this(new DenseMatrix(input_size, hidden_size), new DenseMatrix(hidden_size, hidden_size),
				new DenseVector(hidden_size), bptt_size, non);
	}

	/**
	 * Sutskever, I. (2013). Training Recurrent neural Networks. PhD thesis.
	 * University of Toronto.
	 * 
	 * http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
	 * 
	 * https://gist.github.com/karpathy/d4dee566867f8291f086
	 * 
	 * 
	 * @param I
	 * @return
	 */

	@Override
	public Object backward(Object I) {
		this.dH = (DenseTensor) I;

		DenseTensor dH = (DenseTensor) I;
		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(dH.size());

		if (dh_raw.size() == 0) {
			dh_raw = new DenseVector(hidden_size);
			dh_raw2 = new DenseVector(hidden_size);
			dh_next = new DenseVector(hidden_size);
		}

		{
			int size = dH.sizeOfInnerVectors();
			VectorUtils.enlarge(tmp_dX, size, input_size);
			VectorUtils.enlarge(tmp_dZ, size, hidden_size);
		}

		int start = 0;

		for (int i = 0; i < dH.size(); i++) {
			DenseMatrix dHm = dH.get(i);
			DenseMatrix Hm = H.get(i);
			DenseMatrix Xm = X.get(i);

			DenseMatrix dXm = tmp_dX.subMatrix(start, dHm.rowSize());
			DenseMatrix dZm = tmp_dZ.subMatrix(start, dHm.rowSize());

			dXm.setAll(0);
			dZm.setAll(0);

			start += dHm.rowSize();

			non.backward(Hm, dZm);

			int seq_len = Hm.rowSize();

			dh_next.setAll(0);
			h0.setAll(0);

			boolean is_long_seq = false;

			if (seq_len > 10) {
				is_long_seq = true;
			}

			for (int j = seq_len - 1; j >= 0; j--) {
				DenseVector dh = dHm.row(j);

				VectorMath.add(dh_next, dh);

				VectorMath.multiply(dh, dZm.row(j), dh_raw);

				VectorMath.productRows(dh_raw.toDenseMatrix(), Whh, dh_next.toDenseMatrix(), false);

				if (bptt_size == 1) {
					DenseVector h_prev = j == 0 ? h0 : Hm.row(j - 1);
					VectorMath.outerProduct(h_prev, dh_raw, dWhh, true);

					DenseVector x = Xm.row(j);
					VectorMath.outerProduct(x, dh_raw, dWxh, true);

					DenseVector dx = dXm.row(j);
					VectorMath.productRows(dh_raw.toDenseMatrix(), Wxh, dx.toDenseMatrix(), true);

					VectorMath.add(dh_raw, db);
				} else {
					VectorUtils.copy(dh_raw, dh_raw2);

					int size = Math.max(0, j - bptt_size);

					for (int k = j; k >= size; k--) {
						DenseVector h_prev = k == 0 ? h0 : Hm.row(k - 1);
						VectorMath.outerProduct(h_prev, dh_raw2, dWhh, true);

						DenseVector x = Xm.row(k);
						VectorMath.outerProduct(x, dh_raw2, dWxh, true);

						DenseVector dx = dXm.row(k);
						VectorMath.productRows(dh_raw2.toDenseMatrix(), Wxh, dx.toDenseMatrix(), true);

						VectorMath.productRows(dh_raw2.toDenseMatrix(), Whh, dh_raw2.toDenseMatrix(), false);

						VectorMath.multiply(dh_raw2, dZm.row(k), dh_raw2);

						if (is_long_seq) {
							double norm = VectorMath.normL2(dh_raw2);
							System.out.printf("%d: %f\n", k, norm);
						}

						VectorMath.add(dh_raw2, db);
					}

					if (is_long_seq) {
						System.out.println();
					}
				}
			}

			dX.add(dXm);
		}

		this.dX = dX;

		return dX;
	}

	@Override
	public Layer copy() {
		return new RnnLayer(Wxh, Whh, b, bptt_size, non);
	}

	@Override
	public DenseTensor forward(Object I) {
		this.X = (DenseTensor) I;

		DenseTensor X = (DenseTensor) I;
		DenseTensor H = new DenseTensor();
		H.ensureCapacity(X.size());

		if (h0.size() == 0) {
			h0 = new DenseVector(hidden_size);
		}

		VectorUtils.enlarge(tmp_H, X.sizeOfInnerVectors(), hidden_size);
		int start = 0;

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix Hm = tmp_H.subMatrix(start, Xm.rowSize());
			Hm.setAll(0);

			start += Xm.rowSize();

			h0.setAll(0);

			for (int t = 0; t < Xm.rowSize(); t++) {
				DenseVector h = Hm.row(t);

				DenseVector h_prev = t == 0 ? h0 : Hm.row(t - 1);

				VectorMath.product(h_prev, Whh, h, false);

				DenseVector x = Xm.row(t);

				VectorMath.product(x, Wxh, h, true);

				VectorMath.add(b, h);

				non.forward(h.toDenseMatrix(), h.toDenseMatrix());
			}
			H.add(Hm);
		}

		this.H = H;

		return H;
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
		ret.add(dWxh);
		ret.add(dWhh);
		return ret;
	}

	public DenseVector getH0() {
		return h0;
	}

	@Override
	public int getInputSize() {
		return Wxh.rowSize();
	}

	public Nonlinearity getNonlinearity() {
		return non;
	}

	@Override
	public int getOutputSize() {
		return hidden_size;
	}

	@Override
	public DenseTensor getW() {
		DenseTensor ret = new DenseTensor();
		ret.add(Wxh);
		ret.add(Whh);
		return ret;
	}

	@Override
	public void init() {
		ParameterInitializer.init2(Wxh);
		VectorMath.identity(Whh, 1);
	}

	@Override
	public void prepare() {
		dWxh = Wxh.copy(true);
		dWhh = Whh.copy(true);
		db = b.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		Wxh = new DenseMatrix(ois);
		Whh = new DenseMatrix(ois);
		b = new DenseVector(ois);
		Class c = Class.forName(ois.readUTF());
		non = (Nonlinearity) c.newInstance();

		this.input_size = Wxh.rowSize();
		this.hidden_size = Wxh.colSize();
	}

	public void resetH0() {
		h0.setAll(0);
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		Wxh.writeObject(oos);
		Whh.writeObject(oos);
		b.writeObject(oos);
		oos.writeUTF(non.getClass().getName());

	}

}
