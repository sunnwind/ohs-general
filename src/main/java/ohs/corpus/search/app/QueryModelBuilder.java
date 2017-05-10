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

	public SparseVector build1(SparseVector Q) throws Exception {
		DenseMatrix m = new DenseMatrix(Q.size());
		CounterMap<String, String> cm = Generics.newCounterMap();

		for (int j = 0; j < Q.size(); j++) {
			int w1 = Q.indexAt(j);
			double pr1 = vocab.getProb(w1);
			double idf1 = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w1));
			DenseVector e1 = E.row(w1);

			for (int k = j + 1; k < Q.size(); k++) {
				int w2 = Q.indexAt(k);
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

		SparseVector ret = Q.copy();
		ret.normalize();

		ArrayMath.randomWalk(m.values(), ret.values(), 20);

		return ret;
	}

	public SparseVector build2(SparseVector Q) throws Exception {
		DenseMatrix m = new DenseMatrix(Q.size());

		for (int i = 0; i < Q.size(); i++) {
			int w1 = Q.indexAt(i);
			DenseVector e1 = E.row(w1);

			for (int j = i + 1; j < Q.size(); j++) {
				int w2 = Q.indexAt(j);
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

		SparseVector ret = new SparseVector(Q.copyIndexes());

		for (int i = 0; i < Q.size(); i++) {
			int w = Q.indexAt(i);
			double pr_w_in_q = Q.valueAt(i);
			double pr_w_in_q_emb = ArrayMath.dotProduct(Q.values(), m.row(i).values());
			ret.addAt(i, w, pr_w_in_q_emb);
		}

		ArrayMath.randomWalk(m.values(), ret.values(), 20);

		ret.summation();

		return ret;
	}

}
