package ohs.corpus.search.app;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ohs.math.VectorUtils;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.CounterMap;
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class LemmaExpander {

	private Vocab vocab;

	private Map<Integer, Integer> wordToLemma;

	private SetMap<Integer, Integer> lemmaToVars;

	public LemmaExpander(Vocab vocab, Map<String, String> m) throws Exception {
		this.vocab = vocab;
		wordToLemma = Generics.newHashMap(m.size());

		for (Entry<String, String> e : m.entrySet()) {
			String word = e.getKey();
			String lemma = e.getValue();
			int w = vocab.indexOf(word);
			int l = vocab.indexOf(lemma);
			if (w >= 0 && l >= 0) {
				wordToLemma.put(w, l);
			}
		}

		for (int w = 0; w < vocab.size(); w++) {
			if (!wordToLemma.containsKey(w)) {
				wordToLemma.put(w, w);
			}
		}

		lemmaToVars = Generics.newSetMap();

		for (Entry<Integer, Integer> e : wordToLemma.entrySet()) {
			int w = e.getKey();
			int l = e.getValue();
			lemmaToVars.put(l, w);
		}

		lemmaToVars.trimToSize();
	}

	public SparseVector expand(SparseVector Q) {
		SparseMatrix wordToVars = mapWordToVariants(Q);

		IntegerArray idxs = new IntegerArray();
		DoubleArray vals = new DoubleArray();

		for (int i = 0; i < Q.size(); i++) {
			int w = Q.indexAt(i);
			double cnt_w_in_q = Q.valueAt(i);
			SparseVector vars = wordToVars.row(w);

			for (int j = 0; j < vars.size(); j++) {
				int v = vars.indexAt(j);
				double pr_v_in_c = vocab.getProb(v);

				if (w == v) {
					pr_v_in_c *= 5;
				}
				vars.setAt(j, pr_v_in_c);
			}

			vars.normalizeAfterSummation();

			if (vars.size() < 2) {
				idxs.add(w);
				vals.add(cnt_w_in_q);
				// vals.add(vocab.getCount(w));
			} else {
				// double weight = 1f / vars.size();
				// double cnt2 = weight * cnt;

				for (int w2 : vars.indexes()) {
					double pr2 = cnt_w_in_q * vars.value(w2);
					idxs.add(w2);
					vals.add(pr2);
					// vals.add(vocab.getCount(w2));
				}
			}
		}

		idxs.trimToSize();
		vals.trimToSize();

		return new SparseVector(idxs.values(), vals.values());
	}

	public SparseMatrix mapWordToVariants(SparseVector Q) {
		CounterMap<Integer, Integer> cm = Generics.newCounterMap();
		for (int j = 0; j < Q.size(); j++) {
			int w = Q.indexAt(j);
			Integer lemma = wordToLemma.get(w);

			if (lemma == null) {
				continue;
			}

			Set<Integer> vars = lemmaToVars.get(lemma);

			if (vars == null) {
				continue;
			}

			for (int var : vars) {
				cm.incrementCount(w, var, 1);
			}
		}

		// System.out.println(VectorUtils.toCounterMap(cm, vocab, vocab));
		// System.out.println();

		return VectorUtils.toSparseMatrix(cm);
	}

}
