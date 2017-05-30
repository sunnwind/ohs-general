package ohs.ir.search.model;

import ohs.corpus.type.DocumentCollection;
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

	private int num_top_docs = 10;

	private DocumentCollection dc;

	private Vocab vocab;

	private Scorer scorer;

	public DocumentCentralityEstimator(DocumentCollection dc, Scorer scorer) {
		this.dc = dc;
		this.scorer = scorer;
		this.vocab = dc.getVocab();
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

		for (int i = 0; i < scores.size(); i++) {
			int dseq1 = scores.indexAt(i);
			SparseVector dv1 = dvs.rowAt(i);

			for (int j = i + 1; j < scores.size(); j++) {
				int dseq2 = scores.indexAt(j);
				SparseVector dv2 = dvs.rowAt(j);
				double forward_score = scorer.scoreFromCollection(dv1, new SparseVector(new int[] { dseq2 })).valueAt(0);
				double backward_score = scorer.scoreFromCollection(dv2, new SparseVector(new int[] { dseq1 })).valueAt(0);
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

}
