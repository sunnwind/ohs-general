package ohs.corpus.search.model;

import java.util.List;
import java.util.WeakHashMap;

import ohs.corpus.search.app.DocumentSearcher;
import ohs.corpus.search.index.InvertedIndex;
import ohs.corpus.search.index.WordFilter;
import ohs.corpus.type.DocumentCollection;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;

public class PassageScorer extends Scorer {

	private Vocab vocab;

	private DocumentCollection dc;

	private InvertedIndex ii;

	private double dirichlet_prior = 1000;

	private double mixture_jm = 0.7;

	private WordFilter filter;

	private WeakHashMap<Integer, IntegerArrayMatrix> cache = Generics.newWeakHashMap();

	private int len_psg_fix = 500;

	public PassageScorer(DocumentSearcher ds) {
		super(ds);

	}

	public IntegerArrayMatrix getPassages(int docseq) throws Exception {
		IntegerArrayMatrix ret = cache.get(docseq);

		if (ret == null) {
			ret = new IntegerArrayMatrix();
			IntegerArrayMatrix doc = dc.getSents(docseq).getSecond();
			int len_d = doc.sizeOfEntries();
			int loc = 0;
			IntegerArray psg = new IntegerArray(len_psg_fix);

			for (IntegerArray sent : doc) {
				for (int w : sent) {
					psg.add(w);
					loc++;
					if (psg.size() == len_psg_fix || loc == len_d) {
						ret.add(psg);
						psg = new IntegerArray(len_psg_fix);
					}
				}
			}
			cache.put(docseq, ret);
		}

		return ret;
	}

	public List<SparseVector> getPassageVectors(int docseq) throws Exception {
		IntegerArrayMatrix psgs = getPassages(docseq);
		List<SparseVector> ret = Generics.newArrayList(psgs.size());
		for (IntegerArray psg : psgs) {
			Counter<Integer> c = Generics.newCounter(psg.size());
			for (int w : psg) {
				if (filter.filter(w)) {
					continue;
				}
				c.incrementCount(w, 1);
			}
			ret.add(VectorUtils.toSparseVector(c));
		}

		return ret;
	}

	public SparseVector score(SparseVector lm_q, SparseVector docCnts) throws Exception {
		double pr_w_in_c = 0;
		double pr_w_in_psg_jm = 0;
		double pr_w_in_q = 0;
		double cnt_w_in_psg = 0;
		double len_psg = 0;
		double div = 0;
		int w = 0;

		SparseVector ret = new SparseVector(ArrayUtils.copy(docCnts.indexes()));

		for (int i = 0; i < ret.size(); i++) {
			int docseq = ret.indexAt(i);
			List<SparseVector> psgs = getPassageVectors(docseq);
			DoubleArray psgScores = new DoubleArray(psgs.size());

			for (int j = 0; j < psgs.size(); j++) {
				SparseVector psg = psgs.get(j);
				len_psg = psg.sum();
				div = 0;

				for (int k = 0; k < lm_q.size(); k++) {
					w = lm_q.indexAt(k);
					pr_w_in_q = lm_q.probAt(k);
					cnt_w_in_psg = psg.value(w);
					pr_w_in_psg_jm = TermWeighting.twoStageSmoothing(cnt_w_in_psg, len_psg, pr_w_in_c, dirichlet_prior, pr_w_in_c,
							mixture_jm);

					if (pr_w_in_psg_jm > 0) {
						div += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_psg_jm);
					}
				}
				psgScores.add(Math.exp(-div));
			}

			// double score = ArrayMath.max(psgScores.elementData());
			double score = ArrayMath.mean(psgScores.values());
			ret.addAt(i, score);
		}

		return ret;
	}

	public void setDirichletPrior(double dirichlet_prior) {
		this.dirichlet_prior = dirichlet_prior;
	}

	public void setMixtureJM(double mixture_jm) {
		this.mixture_jm = mixture_jm;
	}

}
