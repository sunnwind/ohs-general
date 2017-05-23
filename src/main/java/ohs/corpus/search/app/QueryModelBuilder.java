package ohs.corpus.search.app;

import java.util.List;

import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
import ohs.types.number.DoubleArray;
import ohs.utils.Generics;

public class QueryModelBuilder {

	private Vocab vocab;

	private RandomAccessDenseMatrix E;

	public QueryModelBuilder(Vocab vocab, RandomAccessDenseMatrix E) {
		this.vocab = vocab;
		this.E = E;
	}

	Counter<String> build1(Counter<String> Q, DenseVector e_ext, double idf_ext) throws Exception {
		SparseVector ret = build1(VectorUtils.toSparseVector(Q, vocab), e_ext, idf_ext);
		return VectorUtils.toCounter(ret, vocab);
	}

	public SparseVector build1(SparseVector Q, DenseVector e_ext, double idf_ext) throws Exception {
		List<DenseVector> e_ds = Generics.newArrayList(Q.size() + 2);
		DoubleArray idfs = new DoubleArray(Q.size() + 2);

		DenseVector e_q = new DenseVector(E.colSize());
		double idf_q = 0;
		int cnt = 0;
		for (int w : Q.indexes()) {
			double idf = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w));
			DenseVector e = E.row(w);
			VectorMath.add(e, e_q);
			idf_q += idf;

			if (e != null) {
				cnt++;
			}

			e_ds.add(e);
			idfs.add(idf);
		}

		if (cnt > 0) {
			e_q.multiply(1f / cnt);
			idf_q /= cnt;
		}

		e_ds.add(e_q);
		idfs.add(idf_q);

		if (e_ext != null) {
			e_ds.add(e_ext);
			idfs.add(idf_ext);
		}

		DenseMatrix m = new DenseMatrix(e_ds.size());
		// CounterMap<String, String> cm = Generics.newCounterMap();

		for (int j = 0; j < e_ds.size(); j++) {
			double idf1 = idfs.get(j);
			DenseVector e1 = e_ds.get(j);

			if (e1 == null) {
				continue;
			}

			for (int k = j + 1; k < e_ds.size(); k++) {
				double idf2 = idfs.get(k);
				DenseVector e2 = e_ds.get(k);

				if (e2 == null) {
					continue;
				}

				double cosine = VectorMath.dotProduct(e1, e2);
				double weight = Math.exp(cosine) * idf1 * idf2;
				m.add(j, k, weight);
				m.add(k, j, weight);
				// cm.incrementCount(vocab.getObject(w1), vocab.getObject(w2), weight);
				// cm.incrementCount(vocab.getObject(w2), vocab.getObject(w1), weight);
			}
		}

		m.normalizeColumns();

		SparseVector ret = new SparseVector(e_ds.size());

		// ArrayMath.randomWalk(m.values(), ret.values(), 20);
		ArrayMath.randomWalk(m.values(), ret.values(), 20, 0.0000001, 1);

		ret = ret.subVector(0, Q.size());
		ret.setIndexes(ArrayUtils.copy(Q.indexes()));
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

	public RandomAccessDenseMatrix getEmbeddingMatrix() {
		return E;
	}

}
