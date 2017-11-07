package ohs.ml.neuralnet.cost;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.types.number.IntegerArray;

/**
 * 
 * http://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization
 * 
 * http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
 * 
 * @author ohs
 *
 */
public class CrossEntropyCostFunction implements CostFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5599113065557841190L;

	public static double getL2RegularizationTerm(double reg_lambda, DenseMatrix W, double data_size) {
		return 0.5 * reg_lambda * (1f / data_size) * VectorMath.sumAfterSquared(W);
	}

	private int cor_cnt = 0;

	private double cost = 0;

	private DenseMatrix tmp_D = new DenseMatrix(0);

	public DenseMatrix checkCorrectness(DenseTensor Yh, DenseMatrix Y) {
		DenseMatrix C = Y.copy(true);
		for (int i = 0; i < Yh.size(); i++) {
			DenseMatrix Yhm = Yh.get(i);
			DenseVector Ym = Y.row(i);
			DenseVector Cm = C.row(i);

			for (int j = 0; j < Yhm.rowSize(); j++) {
				DenseVector yhm = Yhm.row(j);

				int pred = yhm.argMax();
				int ans = (int) Ym.value(j);

				if (pred == ans) {
					Cm.add(j, 1);
				}
			}
		}
		return C;
	}

	public DenseMatrix evaluate(DenseMatrix Yh, DenseVector y) {
		cost = 0;
		cor_cnt = 0;

		if (tmp_D == null || tmp_D.rowSize() < Yh.rowSize()) {
			tmp_D = Yh.copy(true);
		}

		DenseMatrix D = tmp_D.subMatrix(Yh.rowSize());

		for (int i = 0; i < Yh.rowSize(); i++) {
			DenseVector yh = Yh.row(i);
			DenseVector d = D.row(i);

			int pred = yh.argMax();
			int ans = (int) y.value(i);

			if (pred == ans) {
				cor_cnt++;
			}

			if (yh.value(ans) != 0) {
				cost += Math.log(yh.value(ans));
			}

			double sum = 0;
			for (int j = 0; j < yh.size(); j++) {
				double v = j == ans ? -1 : 0;
				d.set(j, yh.value(j) + v);
				sum += d.value(j);
			}
			d.setSum(sum);
		}

		cost /= Yh.rowSize() * -1f;
		return D;
	}

	public DenseTensor evaluate(DenseTensor Yh, DenseMatrix Y) {
		cost = 0;
		cor_cnt = 0;

		VectorUtils.enlarge(tmp_D, Yh.sizeOfInnerVectors(), Yh.row(0).colSize());

		DenseTensor D = new DenseTensor();
		D.ensureCapacity(Yh.size());

		int start = 0;

		for (int i = 0; i < Yh.size(); i++) {
			DenseMatrix Yhm = Yh.get(i);
			DenseMatrix Dm = tmp_D.subMatrix(start, Yhm.rowSize());
			Dm.setAll(0);

			start += Yhm.rowSize();

			DenseVector Ym = Y.row(i);

			for (int j = 0; j < Yhm.rowSize(); j++) {
				DenseVector yhm = Yhm.row(j);
				DenseVector dm = Dm.row(j);

				int pred = yhm.argMax();
				int ans = (int) Ym.value(j);

				if (pred == ans) {
					cor_cnt++;
				}

				if (yhm.value(ans) != 0) {
					cost += Math.log(yhm.value(ans));
				}

				VectorUtils.copy(yhm, dm);
				dm.add(ans, -1);

				// for (int k = 0; k < yhm.size(); k++) {
				// double v = k == ans ? -1 : 0;
				// dm.set(k, yhm.value(k) + v);
				// }
				// dm.summation();
			}
			D.add(Dm);
		}

		cost /= Yh.sizeOfInnerVectors() * -1d;
		return D;
	}

	@Override
	public int getCorrectCount() {
		return cor_cnt;
	}

	@Override
	public double getCost() {
		return cost;
	}

	public double getCost(double reg_lambda, DenseMatrix W, int data_size) {
		return cost + getL2RegularizationTerm(reg_lambda, W, data_size);
	}

}
