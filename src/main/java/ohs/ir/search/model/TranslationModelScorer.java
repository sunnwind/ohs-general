package ohs.ir.search.model;

import java.util.Map.Entry;

import ohs.corpus.type.DocumentCollection;
import ohs.ir.search.app.WordSearcher;
import ohs.ir.search.index.InvertedIndex;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;

public class TranslationModelScorer extends Scorer {

	private Vocab vocab;

	private DocumentCollection dc;

	private TranslationModelBuilder dmb;

	private double mixture_jm = 0.5;

	private double prior_dir = 2000;

	private double mixture_trsm_sem = 0;

	private SparseMatrix E;

	public TranslationModelScorer(Vocab vocab, DocumentCollection dc, InvertedIndex ii, WordSearcher ws, WordFilter wf) throws Exception {
		super(vocab, dc, ii);

		dmb = new TranslationModelBuilder(vocab, dc, ws, wf);
	}

	public double score(SparseVector lm_q, int dseq) throws Exception {
		SparseVector dv = dc.getDocVector(dseq);
		SparseMatrix T1 = dmb.getTranslationModel(dseq);

		CounterMap<Integer, Integer> cm = Generics.newCounterMap();

		for (int i = 0; i < dv.size(); i++) {
			int w1 = dv.indexAt(i);

			SparseVector row = E.row(w1);

			if (row.size() == 0) {
				continue;
			}

			for (int j = i + 1; j < dv.size(); j++) {
				int w2 = dv.indexAt(j);
				double cosine = row.value(w2);

				if (cosine > 0) {
					cm.setCount(w1, w2, Math.exp(10 * cosine));
				}
			}
		}

		SparseMatrix T2 = dmb.getTranslationModel(cm, dv);

		// System.out.println(VectorUtils.toCounterMap(cm1, vocab, vocab));
		// System.out.println(VectorUtils.toCounterMap(cm2, vocab, vocab));
		// System.out.println();

		// if (mixture_trsm_sem > 0) {
		// SparseMatrix T2 =
		// dmb.getTranslationModel(dmb.getSemanticProximities(dseq), dv);
		// SparseMatrix T3 = VectorMath.addAfterMultiply(T, 1 -
		// mixture_trsm_sem, T2, mixture_trsm_sem);
		// T3.normalizeColumns();
		// T = T3;
		// }

		mixture_trsm_sem = 0.5;

		// T1 = VectorMath.addAfterMultiply(T1, 1 - mixture_trsm_sem, T2,
		// mixture_trsm_sem);
		// T1.normalizeColumns();

		SparseVector lm_d = dv.copy();
		lm_d.normalize();

		double div = 0;

		for (int i = 0; i < lm_q.size(); i++) {
			int w = lm_q.indexAt(i);
			double pr_w_in_q = lm_q.valueAt(i);
			SparseVector trsm = T1.row(w);
			double pr_w_in_d_tr = VectorMath.dotProduct(trsm, lm_d);
			double pr_w_in_c = vocab.getProb(w);
			double pr_w_in_d_jm = TermWeighting.jelinekMercerSmoothing(pr_w_in_d_tr, pr_w_in_c, mixture_jm);

			if (pr_w_in_d_jm > 0) {
				div += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d_jm);
			}
		}

		double score = Math.exp(-div);
		return score;
	}

	public double score0(SparseVector lm_q, int dseq) throws Exception {
		SparseVector dv = dc.getDocVector(dseq);

		CounterMap<Integer, Integer> cm1 = dmb.getProximities(dseq);
		SparseMatrix T = dmb.getTranslationModel(cm1, dv);

		SparseVector lm_d = dv.copy();
		lm_d.normalize();

		double div = 0;

		for (int i = 0; i < lm_q.size(); i++) {
			int w = lm_q.indexAt(i);
			double pr_w_in_q = lm_q.valueAt(i);
			SparseVector trsm = T.row(w);
			double pr_w_in_d_tr = VectorMath.dotProduct(trsm, lm_d);
			double pr_w_in_c = vocab.getProb(w);
			double pr_w_in_d_jm = TermWeighting.jelinekMercerSmoothing(pr_w_in_d_tr, pr_w_in_c, mixture_jm);

			if (pr_w_in_d_jm > 0) {
				div += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d_jm);
			}
		}

		double score = Math.exp(-div);
		return score;
	}

	public double score1(SparseVector lm_q, int dseq) throws Exception {
		SparseVector dv = dc.getDocVector(dseq);

		CounterMap<Integer, Integer> cm1 = dmb.getProximities(dseq);
		CounterMap<Integer, Integer> cm2 = dmb.symmetric(cm1);

		DenseVector scores = new DenseVector(3);
		DenseVector weights = new DenseVector(new double[] { 80, 10, 10 });
		weights.normalizeAfterSummation();

		double len_d = dv.sum();

		for (int i = 0; i < lm_q.size(); i++) {
			int w1 = lm_q.indexAt(i);
			double cnt_w1_in_d = dv.value(w1);
			double pr_w1_in_c = vocab.getProb(w1);
			double pr_w1_in_d = TermWeighting.twoStageSmoothing(cnt_w1_in_d, len_d, pr_w1_in_c, prior_dir, pr_w1_in_c, mixture_jm);
			scores.add(0, pr_w1_in_d);

			for (int j = i + 1; j < lm_q.size(); j++) {
				int w2 = lm_q.indexAt(j);
				double cnt_w2_in_d = dv.value(w2);
				double pr_w2_in_c = vocab.getProb(w2);
				double pr_w2_in_d = TermWeighting.twoStageSmoothing(cnt_w2_in_d, len_d, pr_w2_in_c, prior_dir, pr_w2_in_c, mixture_jm);

				double prox_forward = cm1.getCount(w1, w2);
				double prox_bacward = cm2.getCount(w2, w1);

				double score1 = prox_forward * pr_w1_in_d * pr_w2_in_d;
				double score2 = prox_bacward * pr_w1_in_d * pr_w2_in_d;

				scores.add(1, score1);
				scores.add(2, score2);
			}
		}

		double ret = VectorMath.cosine(scores, weights);
		return ret;
	}

	@Override
	public SparseVector scoreFromCollection(SparseVector Q, SparseVector docs) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SparseVector scoreFromIndex(SparseVector lm_q, SparseVector docCnt) throws Exception {
		SparseVector ret = new SparseVector(docCnt.size());
		for (int i = 0; i < docCnt.size(); i++) {
			int dseq = docCnt.indexAt(i);
			double cnt = docCnt.valueAt(i);
			ret.addAt(i, dseq, score0(lm_q, dseq));
		}
		// ret.normalize();

		// ArrayMath.multiply(ret.values(), docCnt.values(), ret.values());
		// ret.summation();

		ret.sortValues();
		return ret;
	}

	public double scoreOld(SparseVector lm_q, int dseq) throws Exception {
		SparseVector dv = dc.getDocVector(dseq);

		CounterMap<Integer, Integer> cm1 = dmb.getProximities(dseq);
		CounterMap<Integer, Integer> cm2 = Generics.newCounterMap();

		for (int w1 : cm1.keySet()) {
			Counter<Integer> c = cm1.getCounter(w1);
			SparseVector sims = E.row(w1);

			for (Entry<Integer, Double> e : c.entrySet()) {
				int w2 = e.getKey();
				double prox = e.getValue();
				double cosine = sims.value(w2);
				double prox2 = prox * Math.exp(10 * cosine);
				cm2.setCount(w1, w2, prox2);
			}
		}

		SparseMatrix T = dmb.getTranslationModel(cm1, dv);

		// if (mixture_trsm_sem > 0) {
		// SparseMatrix T2 =
		// dmb.getTranslationModel2(dmb.getSemanticProximities(dseq), dv);
		// SparseMatrix T3 = VectorMath.addAfterMultiply(T, 1 -
		// mixture_trsm_sem, T2, mixture_trsm_sem);
		// T3.normalizeColumns();
		// T = T3;
		// }

		SparseVector lm_d = dv.copy();
		lm_d.normalize();

		double div = 0;

		for (int i = 0; i < lm_q.size(); i++) {
			int w = lm_q.indexAt(i);
			double pr_w_in_q = lm_q.valueAt(i);
			SparseVector trsm = T.row(w);
			double pr_w_in_d_tr = VectorMath.dotProduct(trsm, lm_d);
			double pr_w_in_c = vocab.getProb(w);
			double pr_w_in_d_jm = TermWeighting.jelinekMercerSmoothing(pr_w_in_d_tr, pr_w_in_c, mixture_jm);

			if (pr_w_in_d_jm > 0) {
				div += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d_jm);
			}
		}

		double score = Math.exp(-div);
		return score;
	}

	public void setMixtureJM(double mixture_jm) {
		this.mixture_jm = mixture_jm;
	}

	public void setMixtureTrsmSem(double mixture_trsm_sem) {
		this.mixture_trsm_sem = mixture_trsm_sem;
	}

	public void setSematicSims(SparseMatrix E) {
		this.E = E;
	}

}
