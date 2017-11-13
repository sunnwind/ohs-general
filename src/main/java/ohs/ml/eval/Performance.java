package ohs.ml.eval;

import java.text.NumberFormat;

import ohs.math.ArrayMath;
import ohs.types.generic.Indexer;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;

public class Performance {

	private Indexer<String> labelIdxer;

	private IntegerArray ans_cnts;

	private IntegerArray cor_cnts;

	private IntegerArray pred_cnts;

	private DoubleArray precisons;

	private DoubleArray recalls;

	private DoubleArray f1;

	private double macro_f1;

	private double micro_f1;

	private double micro_pr;

	private double micro_rc;

	private int total_ans_cnt;

	private int total_cor_cnt;

	private int total_pred_cnt;

	public Performance(int label_size) {
		this(new IntegerArray(label_size), new IntegerArray(label_size), new IntegerArray(label_size));
	}

	public Performance(IntegerArray ans_cnts, IntegerArray pred_cnts, IntegerArray cor_cnts) {
		this(ans_cnts, pred_cnts, cor_cnts, null);
	}

	public Performance(IntegerArray ans_cnts, IntegerArray pred_cnts, IntegerArray cor_cnts,
			Indexer<String> labelIdxer) {
		this.ans_cnts = ans_cnts;
		this.pred_cnts = pred_cnts;
		this.cor_cnts = cor_cnts;
		this.labelIdxer = labelIdxer;

		precisons = new DoubleArray(ans_cnts.size());
		recalls = new DoubleArray(ans_cnts.size());
		f1 = new DoubleArray(ans_cnts.size());

		evaluate();
	}

	public void evaluate() {
		for (int i = 0; i < ans_cnts.size(); i++) {
			precisons.add(Metrics.precision(cor_cnts.get(i), pred_cnts.get(i)));
			recalls.add(Metrics.recall(cor_cnts.get(i), ans_cnts.get(i)));
			f1.add(Metrics.f1(precisons.get(i), recalls.get(i)));
		}

		total_ans_cnt = ArrayMath.sum(ans_cnts.values());
		total_pred_cnt = ArrayMath.sum(pred_cnts.values());
		total_cor_cnt = ArrayMath.sum(cor_cnts.values());

		micro_pr = Metrics.precision(total_cor_cnt, total_pred_cnt);
		micro_rc = Metrics.recall(total_cor_cnt, total_ans_cnt);
		micro_f1 = Metrics.f1(micro_pr, micro_rc);
		macro_f1 = ArrayMath.mean(f1.values());

	}

	public IntegerArray getAnswerCounts() {
		return ans_cnts;
	}

	public IntegerArray getCorrectCounts() {
		return cor_cnts;
	}

	public IntegerArray getPredictCounts() {
		return pred_cnts;
	}

	public void setAnswerCounts(IntegerArray ans_cnts) {
		this.ans_cnts = ans_cnts;
	}

	public void setCorrectCounts(IntegerArray cor_cnts) {
		this.cor_cnts = cor_cnts;
	}

	public void SetLabelIndexer(Indexer<String> labelIdxer) {
		this.labelIdxer = labelIdxer;
	}

	public void setPredictCounts(IntegerArray pred_cnts) {
		this.pred_cnts = pred_cnts;
	}

	public String toString() {
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMinimumFractionDigits(4);
		StringBuffer sb = new StringBuffer();
		sb.append("Label\tAns\tPreds\tCors\tPR\tRC\tF1");

		for (int i = 0; i < ans_cnts.size(); i++) {
			int ans_cnt = ans_cnts.get(i);
			int pred_cnt = pred_cnts.get(i);
			int correct = cor_cnts.get(i);
			double pr = precisons.get(i);
			double rc = recalls.get(i);
			double f = f1.get(i);
			String label = labelIdxer == null ? null : labelIdxer.getObject(i);

			sb.append(String.format("\n%s\t%d\t%d\t%d\t%s\t%s\t%s", label == null ? i + "" : label, ans_cnt, pred_cnt,
					correct, nf.format(pr), nf.format(rc), nf.format(f)));
		}

		// total_ans_cnt = ArrayMath.sum(ans_cnts);
		// total_pred_cnt = ArrayMath.sum(pred_cnts);
		// total_cor_cnt = ArrayMath.sum(cor_cnts);
		//
		// micro_pr = Metrics.precision(total_cor_cnt, total_pred_cnt);
		// micro_rc = Metrics.recall(total_cor_cnt, total_ans_cnt);
		// micro_f1 = Metrics.f1(micro_pr, micro_rc);

		sb.append(String.format("\nTotal\t%d\t%d\t%d\t%s\t%s\t%s", total_ans_cnt, total_pred_cnt, total_cor_cnt,
				nf.format(micro_pr), nf.format(micro_rc), nf.format(micro_f1)));

		return sb.toString();
	}

}