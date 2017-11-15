package ohs.ml.eval;

import java.text.NumberFormat;

import ohs.math.ArrayMath;
import ohs.matrix.DenseVector;
import ohs.types.generic.Indexer;

public class Performance {

	private Indexer<String> labelIdxer;

	private DenseVector ansCnts;

	private DenseVector corCnts;

	private DenseVector predCnts;

	private DenseVector precisons;

	private DenseVector recalls;

	private DenseVector f1;

	private double macro_f1;

	private double micro_f1;

	private double micro_pr;

	private double micro_rc;

	public Performance(DenseVector ans_cnts, DenseVector pred_cnts, DenseVector cor_cnts) {
		this(ans_cnts, pred_cnts, cor_cnts, null);
	}

	public Performance(DenseVector ansCnts, DenseVector predCnts, DenseVector corCnts, Indexer<String> labelIdxer) {
		this.ansCnts = ansCnts;
		this.predCnts = predCnts;
		this.corCnts = corCnts;
		this.labelIdxer = labelIdxer;

		precisons = new DenseVector(ansCnts.size());
		recalls = new DenseVector(ansCnts.size());
		f1 = new DenseVector(ansCnts.size());

		evaluate();
	}

	public Performance(int label_size) {
		this(new DenseVector(label_size), new DenseVector(label_size), new DenseVector(label_size));
	}

	public void evaluate() {
		for (int i = 0; i < ansCnts.size(); i++) {
			precisons.add(i, Metrics.precision(corCnts.value(i), predCnts.value(i)));
			recalls.add(i, Metrics.recall(corCnts.value(i), ansCnts.value(i)));
			f1.add(i, Metrics.f1(precisons.value(i), recalls.value(i)));
		}

		micro_pr = Metrics.precision(corCnts.sum(), predCnts.sum());
		micro_rc = Metrics.recall(corCnts.sum(), ansCnts.sum());
		micro_f1 = Metrics.f1(micro_pr, micro_rc);
		macro_f1 = ArrayMath.mean(f1.values());
	}

	public DenseVector getAnswerCounts() {
		return ansCnts;
	}

	public DenseVector getCorrectCounts() {
		return corCnts;
	}

	public double getMicroF1() {
		return micro_f1;
	}

	public DenseVector getPredictCounts() {
		return predCnts;
	}

	public void setAnswerCounts(DenseVector ans_cnts) {
		this.ansCnts = ans_cnts;
	}

	public void setCorrectCounts(DenseVector cor_cnts) {
		this.corCnts = cor_cnts;
	}

	public void SetLabelIndexer(Indexer<String> labelIdxer) {
		this.labelIdxer = labelIdxer;
	}

	public void setPredictCounts(DenseVector pred_cnts) {
		this.predCnts = pred_cnts;
	}

	@Override
	public String toString() {
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMinimumFractionDigits(4);
		StringBuffer sb = new StringBuffer();
		sb.append("Label\tAns\tPreds\tCors\tPR\tRC\tF1");

		for (int i = 0; i < ansCnts.size(); i++) {
			int ans_cnt = (int) ansCnts.value(i);
			int pred_cnt = (int) predCnts.value(i);
			int correct = (int) corCnts.value(i);
			double pr = precisons.value(i);
			double rc = recalls.value(i);
			double f = f1.value(i);
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

		sb.append(String.format("\nTotal\t%d\t%d\t%d\t%s\t%s\t%s", (int) ansCnts.sum(), (int) predCnts.sum(),
				(int) corCnts.sum(), nf.format(micro_pr), nf.format(micro_rc), nf.format(micro_f1)));

		return sb.toString();
	}

}