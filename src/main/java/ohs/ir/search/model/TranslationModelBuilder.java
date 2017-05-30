package ohs.ir.search.model;

import java.util.List;

import ohs.corpus.type.DocumentCollection;
import ohs.ir.search.app.WordSearcher;
import ohs.ir.search.index.WordFilter;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;

public class TranslationModelBuilder {

	private Vocab vocab;

	private DocumentCollection dc;

	private WordFilter filter;

	private WordSearcher ws;

	private double prior_dir = 2000;

	private double mixture_jm = 0.5;

	private double mixture_self_trs = 0;

	private int window_size = 3;

	public TranslationModelBuilder(Vocab vocab, DocumentCollection ldc, WordSearcher ws, WordFilter filter) throws Exception {
		this.vocab = vocab;
		this.dc = ldc;
		this.filter = filter;
		this.ws = ws;
	}

	public SparseVector buildTranslatedModel(int dseq) throws Exception {
		return buildTranslatedModel(dc.getSents(dseq).getSecond(), dc.getDocVector(dseq));
	}

	public SparseVector buildTranslatedModel(IntegerArrayMatrix doc, SparseVector dv) throws Exception {
		SparseMatrix T = getTranslationModel(getProximities(doc), dv);
		SparseVector lm_d = dv.copy();
		lm_d.normalize();
		return translateModel(T, lm_d);
	}

	public SparseVector buildTranslatedModel(SparseMatrix T, SparseVector dv) {
		SparseVector lm_d = dv.copy();
		lm_d.normalize();
		return translateModel(T, lm_d);
	}

	public SparseMatrix buildTranslatedModels(int[] docseqs) throws Exception {
		List<Integer> idxs = Generics.newArrayList(docseqs.length);
		List<SparseVector> rows = Generics.newArrayList(docseqs.length);
		for (int i = 0; i < docseqs.length; i++) {
			int docseq = docseqs[i];
			SparseVector lm_d = buildTranslatedModel(docseq);
			idxs.add(docseq);
			rows.add(lm_d);
		}
		return new SparseMatrix(idxs, rows);
	}

	public SparseMatrix buildTranslatedModels(SparseMatrix dvs) throws Exception {
		List<Integer> idxs = Generics.newArrayList(dvs.size());
		List<SparseVector> rows = Generics.newArrayList(dvs.size());
		for (int i = 0; i < dvs.size(); i++) {
			int docseq = dvs.indexAt(i);
			SparseVector lm_d = buildTranslatedModel(dc.getSents(docseq).getSecond(), dvs.rowAt(i));
			idxs.add(docseq);
			rows.add(lm_d);
		}
		return new SparseMatrix(idxs, rows);
	}

	public CounterMap<Integer, Integer> getProximities(int docseq) throws Exception {
		return getProximities(dc.getSents(docseq).getSecond());
	}

	public CounterMap<Integer, Integer> getProximities(int[] docseqs) throws Exception {
		CounterMap<Integer, Integer> ret = Generics.newCounterMap();
		for (int docseq : docseqs) {
			ret.incrementAll(getProximities(docseq));
		}
		return ret;
	}

	public CounterMap<Integer, Integer> getProximities(IntegerArrayMatrix doc) throws Exception {
		CounterMap<Integer, Integer> cm = Generics.newCounterMap();

		for (IntegerArray sent : doc) {
			for (int i = 0; i < sent.size(); i++) {
				int w1 = sent.get(i);

				if (filter.filter(w1)) {
					continue;
				}

				int s = i + 1;
				int e = Math.min(sent.size(), s + window_size);

				for (int j = s; j < e; j++) {
					int w2 = sent.get(j);

					if (filter.filter(w2)) {
						continue;
					}

					double dist = j - i;
					double score = 1f / dist;
					cm.incrementCount(w1, w2, score);
				}
			}
		}
		return cm;
	}

	public CounterMap<Integer, Integer> getSemanticProximities(CounterMap<Integer, Integer> proxs, SparseVector dv) throws Exception {
		CounterMap<Integer, Integer> cm = Generics.newCounterMap();
		for (int i = 0; i < dv.size(); i++) {
			int w1 = dv.indexAt(i);
			for (int j = i + 1; j < dv.size(); j++) {
				int w2 = dv.indexAt(j);
				if (proxs.getCount(w1, w2) > 0) {
					double cosine = ws.getCosine(w1, w2);
					if (cosine > 0) {
						cm.setCount(w1, w2, cosine);
					}
				}
			}
		}
		return cm;
	}

	public CounterMap<Integer, Integer> getSemanticProximities(int dseq) throws Exception {
		return getSemanticProximities(getProximities(dseq), dc.getDocVector(dseq));
	}

	public SparseMatrix getTranslationModel(CounterMap<Integer, Integer> cm, SparseVector dv) throws Exception {
		for (int w1 : cm.keySet()) {
			Counter<Integer> c = cm.getCounter(w1);
			for (int w2 : c.keySet()) {
				double prox = c.getCount(w2);
				// double pr_w_in_d = TermWeighting.twoStageSmoothing(w2, dv,
				// vocab, prior_dir, mixture_jm);
				double pr_w_in_d = 1;
				cm.setCount(w1, w2, prox * pr_w_in_d);
			}
		}

		cm = symmetric(cm);

		SparseMatrix T = VectorUtils.toSparseMatrix(cm);
		T.normalizeColumns();

		if (mixture_self_trs > 0) {
			for (int i = 0; i < T.rowSize(); i++) {
				int w1 = T.indexAt(i);
				SparseVector trm_w = T.rowAt(i);

				for (int j = 0; j < trm_w.size(); j++) {
					int w2 = trm_w.indexAt(j);
					double pr_w1_to_w2 = trm_w.valueAt(j);

					if (w1 == w2) {
						pr_w1_to_w2 = mixture_self_trs + (1 - mixture_self_trs) * pr_w1_to_w2;
					} else {
						pr_w1_to_w2 = (1 - mixture_self_trs) * pr_w1_to_w2;
					}
					trm_w.setAt(j, pr_w1_to_w2);
				}
			}
			T.normalizeColumns();
		}

		return T;
	}

	public SparseMatrix getTranslationModel(int docseq) throws Exception {
		return getTranslationModel(getProximities(docseq), dc.getDocVector(docseq));
	}

	public SparseMatrix getTranslationModel2(CounterMap<Integer, Integer> cm, SparseVector dv) throws Exception {
		cm = symmetric(cm);
		SparseMatrix T = VectorUtils.toSparseMatrix(cm);
		T.normalizeColumns();
		return T;
	}

	public void setDirichletPrior(double prior_dir) {
		this.prior_dir = prior_dir;
	}

	public void setMixtureJM(double mixture_jm) {
		this.mixture_jm = mixture_jm;
	}

	public void setMixtureSelfTransition(double mixture_self_trs) {
		this.mixture_self_trs = mixture_self_trs;
	}

	public void setWindowSize(int window_size) {
		this.window_size = window_size;
	}

	public CounterMap<Integer, Integer> symmetric(CounterMap<Integer, Integer> cm) {
		CounterMap<Integer, Integer> cm2 = Generics.newCounterMap();
		for (int w1 : cm.keySet()) {
			Counter<Integer> c = cm.getCounter(w1);
			for (int w2 : c.keySet()) {
				double value = c.getCount(w2);
				if (w1 == w2) {
					cm2.incrementCount(w1, w2, value);
				} else {
					cm2.incrementCount(w1, w2, value);
					cm2.incrementCount(w2, w1, value);
				}
			}
		}
		return cm2;
	}

	public SparseVector translateModel(SparseMatrix T, SparseVector lm_d) {
		SparseVector ret = new SparseVector(lm_d.size());
		for (int i = 0; i < lm_d.size(); i++) {
			int w = lm_d.indexAt(i);
			SparseVector trsm = T.row(w);
			ret.addAt(i, w, VectorMath.dotProduct(trsm, lm_d));
		}
		return ret;
	}

}
