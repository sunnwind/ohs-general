package ohs.corpus.search.app;

import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;

public class QueryModelBuilder {

	private Vocab vocab;

	private RandomAccessDenseMatrix E;

	public QueryModelBuilder(Vocab vocab, RandomAccessDenseMatrix E) {
		this.vocab = vocab;
		this.E = E;
	}

	public SparseVector estimate(SparseVector lm_q) throws Exception {
		DenseMatrix m = new DenseMatrix(lm_q.size());
		CounterMap<String, String> cm = Generics.newCounterMap();

		for (int j = 0; j < lm_q.size(); j++) {
			int w1 = lm_q.indexAt(j);
			double pr1 = vocab.getProb(w1);
			double idf1 = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w1));
			DenseVector e1 = E.row(w1);

			for (int k = j + 1; k < lm_q.size(); k++) {
				int w2 = lm_q.indexAt(k);
				double pr2 = vocab.getProb(w2);
				double idf2 = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w2));

				DenseVector e2 = E.row(w2);
				double cosine = VectorMath.dotProduct(e1, e2);
				double weight = Math.exp(cosine);
				m.add(j, k, weight);
				m.add(k, j, weight);
				cm.incrementCount(vocab.getObject(w1), vocab.getObject(w2), weight);
				cm.incrementCount(vocab.getObject(w2), vocab.getObject(w1), weight);
			}
		}

		m.normalizeColumns();

		SparseVector lm_q2 = lm_q.copy();

		ArrayMath.randomWalk(m.values(), lm_q2.values(), 20);

		return lm_q2;
	}

	public SparseVector estimate2(SparseVector lm_q) throws Exception {
		DenseMatrix m = new DenseMatrix(lm_q.size());

		for (int i = 0; i < lm_q.size(); i++) {
			int w1 = lm_q.indexAt(i);
			DenseVector e1 = E.row(w1);

			for (int j = i + 1; j < lm_q.size(); j++) {
				int w2 = lm_q.indexAt(j);
				DenseVector e2 = E.row(w2);
				double cosine = VectorMath.dotProduct(e1, e2);
				// double weight = CommonMath.sigmoid(cosine);
				// double weight = Math.max(0, cosine);
				double weight = Math.exp(cosine);

				m.add(i, j, weight);
				m.add(j, i, weight);
			}
		}

		m.normalizeColumns();

		SparseVector lm_q2 = new SparseVector(lm_q.copyIndexes());

		for (int i = 0; i < lm_q.size(); i++) {
			int w = lm_q.indexAt(i);
			double pr_w_in_q = lm_q.valueAt(i);
			double pr_w_in_q_emb = ArrayMath.dotProduct(lm_q.values(), m.row(i).values());
			lm_q2.addAt(i, w, pr_w_in_q_emb);
		}

		ArrayMath.randomWalk(m.values(), lm_q2.values(), 20);

		lm_q2.summation();

		return lm_q2;
	}

}
