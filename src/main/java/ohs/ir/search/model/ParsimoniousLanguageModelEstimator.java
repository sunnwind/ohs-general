package ohs.ir.search.model;

import java.text.NumberFormat;

import ohs.corpus.type.DocumentCollection;
import ohs.math.ArrayMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;

/**
 * This class provides parsimonious language model (PLM) estimation methods.
 * 
 * PLM assigns high probabilities to topical terms but low probabilities to
 * common terms using EM algorithm. It can be used to select a set of terms and
 * construct a smaller transition matrix.
 * 
 * 
 * 1 . Hiemstra, D., Robertson, S., and Zaragoza, H. 2004. Parsimonious language
 * models for information retrieval. Proceedings of the 27th annual
 * international ACM SIGIR conference on Research and development in information
 * retrieval, ACM, 178??185.
 * 
 * 2. Na, S.-H., Kang, I.-S., and Lee, J.-H. 2007. Parsimonious translation
 * models for information retrieval. Information Processing & Management 43, 1,
 * 121??145.
 * 
 * 
 * @author Heung-Seon Oh
 * 
 */
public class ParsimoniousLanguageModelEstimator {

	public static void main(String args[]) throws Exception {
		System.out.println("process begins");

		System.out.println("probess ends.");
	}

	private DocumentCollection dc;

	private double epsilon = 0.0000001;

	private int max_iters = 100;

	private double mixture_c = 0.1;

	private boolean printLogs;

	public ParsimoniousLanguageModelEstimator(DocumentCollection dc) {
		this.dc = dc;
	}

	public SparseVector estimate(SparseVector dv) {
		SparseVector lm = dv.copy();
		lm.normalize();

		double logLL = ArrayMath.sumLogProbs(lm.values());
		double logLL_old = logLL;

		NumberFormat nf = NumberFormat.getInstance();
		nf.setMinimumFractionDigits(2);
		nf.setGroupingUsed(false);

		System.out.println("<=== start ===>");

		for (int i = 0; i < max_iters; i++) {
			System.out.println(VectorUtils.toCounter(lm, dc.getVocab()));

			SparseVector lm_old = lm.copy();

			for (int j = 0; j < dv.size(); j++) {
				int w = dv.indexAt(j);
				double cnt_w_in_d = dv.valueAt(j);
				double pr_w_in_d = lm.valueAt(j);
				double pr_w_in_c = dc.getVocab().getProb(w);
				double p1 = mixture_c * pr_w_in_d;
				double p2 = mixture_c * pr_w_in_d + (1 - mixture_c) * pr_w_in_c;
				double ratio = p1 / p2;
				// cnt_w_in_d = 1;
				double e = cnt_w_in_d * ratio;
				lm.setAt(j, e);
			}
			lm.normalizeAfterSummation();

			double diff = ArrayMath.sumSquaredDifferences(lm.values(), lm_old.values());

			if (diff < epsilon) {
				break;
			}

			logLL_old = logLL;
		}
		return lm;
	}

	public void setPrintLogs(boolean printLogs) {
		this.printLogs = printLogs;
	}

}
