package ohs.corpus.search.model;

import ohs.corpus.type.DocumentCollection;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Vocab;

/**
 * This class implements centralities of categories.
 * 
 * The standard centralites are computed by PageRank algorithms where a graph over categories are constructed.
 * 
 * 
 * 
 * 
 * 1. Kurland, O. and Lee, L. 2005. PageRank without hyperlinks: structural re-ranking using links induced by language models. Proceedings
 * of the 28th annual international ACM SIGIR conference on Research and development in information retrieval, 306–313.
 * 
 * 
 * 2. Strube, M. and Ponzetto, S.P. 2006. WikiRelate! computing semantic relatedness using wikipedia. proceedings of the 21st national
 * conference on Artificial intelligence - Volume 2, AAAI Press, 1419–1424.
 * 
 * 
 * @author Heung-Seon Oh
 * 
 * 
 */
public class DocumentCentralityEstimator {

	public static void main(String[] args) {
		System.out.println("process begins.");

		System.out.println("process ends.");
	}

	private double dirichlet_prior = 1500;

	private double mixture_jm = 0;

	private int num_top_docs = 10;

	private DocumentCollection dc;

	private Vocab vocab;

	public DocumentCentralityEstimator(Vocab vocab, DocumentCollection ldc) {
		this.vocab = vocab;
		this.dc = ldc;
	}

	public SparseVector estimate(SparseVector scores) throws Exception {
		double[][] trans_probs = getTransitions(scores);

		SparseVector ret = scores.copy();

		double[] cents = ret.values();
		ArrayUtils.setAll(cents, 1f / cents.length);
		ArrayMath.randomWalk(trans_probs, cents, 10, 0.0000001, 0.85);
		ret.summation();

		return ret;
	}

	private double[][] getTransitions(SparseVector scores) throws Exception {
		SparseMatrix dvs = dc.getDocVectors(scores.indexes());

		int size = dvs.rowSize();

		double[][] tran_probs = ArrayMath.matrix(size);

		for (int i = 0; i < size; i++) {
			SparseVector dv1 = dvs.rowAt(i);

			for (int j = i + 1; j < size; j++) {
				SparseVector dv2 = dvs.rowAt(j);
				double forward_score = score(dv1, dv2);
				double backward_score = score(dv2, dv1);

				tran_probs[i][j] = forward_score;
				tran_probs[j][i] = backward_score;
			}
		}

		for (int i = 0; i < size; i++) {
			double[] probs = tran_probs[i];
			int[] indexes = ArrayUtils.rankedIndexes(probs);
			for (int j = num_top_docs; j < size; j++) {
				probs[indexes[j]] = 0;
			}
		}

		ArrayMath.normalizeColumns(tran_probs);
		return tran_probs;
	}

	private double score(SparseVector d, SparseVector q) {
		double div = 0;
		for (int i = 0; i < q.size(); i++) {
			int w = q.indexAt(i);
			double pr_w_in_q = q.prob(w);
			double pr_w_in_c = vocab.getProb(w);

			double cnt_w_in_d = d.value(w);
			double len_d = d.sum();
			double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, dirichlet_prior, pr_w_in_c, mixture_jm);

			if (pr_w_in_d > 0) {
				div += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d);
			}
		}

		div = Math.exp(-div);
		return div;
	}
}
