package ohs.ml.neuralnet.layer;

import java.util.List;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.nonlinearity.Nonlinearity;
import ohs.utils.Generics;

/**
 *
 * 
 * 
 * http://www.existor.com/en/ml-gru.html
 * 
 * @author ohs
 */
public class GruLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2848535801786544922L;

	/**
	 * input to hidden
	 */
	private DenseMatrix Wxh;
	private DenseMatrix Whh;
	private DenseVector bh;

	private DenseMatrix dWxh;
	private DenseMatrix dWhh;
	private DenseVector dbh;

	private Nonlinearity non;

	private int input_size;
	private int hidden_size;

	private DenseMatrix U;
	private DenseMatrix R;

	private DenseMatrix H;
	private DenseMatrix M;
	private DenseMatrix X;

	private DenseVector h0;
	private DenseVector h0_prev;
	private DenseVector a;

	private DenseVector dh_prev;
	private DenseVector di;
	private DenseVector df;
	private DenseVector doo;
	private DenseVector dg;
	private DenseVector da;

	private DenseMatrix tmp_dX;
	private DenseMatrix tmp_U;
	private DenseMatrix tmp_R;

	private DenseMatrix tmp_H;
	private DenseMatrix tmp_M;
	private DenseMatrix tmp_dC;
	private DenseVector tmp;

	private DenseVector a1;

	private DenseVector a2;

	public GruLayer(DenseMatrix Wxh, DenseMatrix Whh, DenseVector bh, Nonlinearity non) {
		super();
		this.Wxh = Wxh;
		this.Whh = Whh;
		this.bh = bh;
		this.non = non;

		input_size = Wxh.rowSize();
		hidden_size = Wxh.colSize() / 3;
	}

	public GruLayer(int input_size, int hidden_size, Nonlinearity non) {
		this(new DenseMatrix(input_size, hidden_size * 3), new DenseMatrix(hidden_size, hidden_size * 3),
				new DenseVector(hidden_size * 3), non);
	}

	@Override
	public Object backward(Object I) {
		return backward2(I);
	}

	public Object backward2(Object IN) {
		DenseMatrix dH = (DenseMatrix) IN;
		int data_size = dH.rowSize();

		if (tmp_dX == null || tmp_dX.rowSize() < data_size) {
			tmp_dX = new DenseMatrix(data_size, input_size);
			tmp_dC = dH.copy(true);
		}

		if (dh_prev == null) {
			dh_prev = h0.copy(true);

			di = h0.copy(true);
			df = h0.copy(true);
			doo = h0.copy(true);
			dg = h0.copy(true);
		}

		dh_prev.setAll(0);

		DenseMatrix dX = tmp_dX.rows(data_size);
		DenseMatrix dC = tmp_dC.rows(data_size);

		dC.setAll(0);

		for (int t = data_size - 1; t >= 0; t--) {
			DenseVector dh = dH.row(t);
			VectorMath.add(dh, dh_prev, dh);

			DenseVector m = M.row(t);

			VectorMath.multiply(m, dh, doo);

		}
		return dX;
	}

	@Override
	public DenseMatrix forward(Object IN) {
		X = (DenseMatrix) IN;
		int data_size = X.rowSize();

		if (tmp_H == null || tmp_H.rowSize() < data_size) {
			tmp_H = new DenseMatrix(data_size, hidden_size);
			tmp_U = tmp_H.copy(true);
			tmp_R = tmp_H.copy(true);
			tmp_M = tmp_H.copy(true);
		}

		if (h0 == null) {
			h0 = new DenseVector(hidden_size);
			h0_prev = h0.copy(true);
			tmp = h0.copy(true);

			a = new DenseVector(hidden_size * 3);
			da = a.copy(true);

			a1 = new DenseVector(hidden_size * 3);
			a2 = a1.copy(true);
		}

		H = tmp_H.rows(data_size);
		M = tmp_M.rows(data_size);
		U = tmp_U.rows(data_size);
		R = tmp_R.rows(data_size);

		for (int t = 0; t < data_size; t++) {
			DenseVector x = X.row(t);
			DenseVector h_prev = t == 0 ? h0 : H.row(t - 1);

			VectorMath.product(x, Wxh, a1, false);
			VectorMath.product(h_prev, Whh, a2, false);

			DenseVector u = U.row(t);
			DenseVector r = R.row(t);
			DenseVector m = M.row(t);

			for (int j = 0; j < hidden_size; j++) {
				int j2 = hidden_size + j;
				int j3 = hidden_size * 2 + j;
				u.set(j, a1.value(j) + a2.value(j) + bh.value(j));
				r.set(j, a1.value(j2) + a2.value(j2) + bh.value(j2));
				m.set(j, a1.value(j3) + bh.value(j3));
				tmp.set(j, a2.value(j3));
			}

			VectorMath.sigmoid(u, u);
			VectorMath.sigmoid(r, r);

			VectorMath.multiply(r, tmp, tmp);
			VectorMath.add(tmp, m);
			VectorMath.tanh(m, m);

			DenseVector h = H.row(t);

			for (int i = 0; i < h.size(); i++) {
				h.set(i, u.value(i) * m.value(i) + (1 - u.value(i)) * h_prev.value(i));
			}
			h.summation();
		}

		VectorUtils.copy(h0, h0_prev);
		VectorUtils.copy(H.row(data_size - 1), h0);
		return H;
	}

	@Override
	public DenseMatrix getB() {
		return bh.toDenseMatrix();
	}

	@Override
	public DenseMatrix getDB() {
		return dbh.toDenseMatrix();
	}

	@Override
	public DenseMatrix getDW() {
		int size = dWxh.rowSize() + dWhh.rowSize();
		List<DenseVector> ws = Generics.newArrayList(size);

		for (DenseVector w : dWxh) {
			ws.add(w);
		}

		for (DenseVector w : dWhh) {
			ws.add(w);
		}
		return new DenseMatrix(ws);
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
	public DenseMatrix getW() {
		int size = Wxh.rowSize() + Whh.rowSize();
		List<DenseVector> ws = Generics.newArrayList(size);

		for (DenseVector w : Wxh) {
			ws.add(w);
		}

		for (DenseVector w : Whh) {
			ws.add(w);
		}

		return new DenseMatrix(ws);
	}

	public DenseMatrix getWhh() {
		return Whh;
	}

	public DenseMatrix getWxh() {
		return Wxh;
	}

	@Override
	public void init() {
		ParameterInitializer.init2(Wxh);
		ParameterInitializer.init2(Whh);

	}

	@Override
	public void prepare() {
		dWxh = Wxh.copy(true);
		dWhh = Whh.copy(true);
		dbh = bh.copy(true);
	}

	public void resetH0() {
		h0.setAll(0);
	}

}
