package ohs.ml.neuralnet.cost;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
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
		return 0.5 * reg_lambda * (1f / data_size) * VectorMath.sumSquared(W);
	}

	private DenseMatrix tmp_D;

	private double cost = 0;

	private int cor_cnt = 0;

	public DenseMatrix evaluate(DenseMatrix Yh, IntegerArray y) {
		cost = 0;
		cor_cnt = 0;

		if (tmp_D == null || tmp_D.rowSize() < Yh.rowSize()) {
			tmp_D = Yh.copy(true);
		}

		DenseMatrix D = tmp_D.rowsAsMatrix(Yh.rowSize());

		for (int i = 0; i < Yh.rowSize(); i++) {
			DenseVector yh = Yh.row(i);
			DenseVector d = D.row(i);

			int pred = yh.argMax();
			int ans = y.get(i);

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

	@Override
	public int getCorrectCnt() {
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
