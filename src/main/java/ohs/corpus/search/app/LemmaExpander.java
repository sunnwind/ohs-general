package ohs.corpus.search.app;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
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

		// lemmaToVars.trimToSize();
	}

	public SparseVector expand(SparseVector Q) {
		Q = Q.copy();
		Q.normalize();

		IntegerArrayMatrix wordToVars = mapWordToVariants(Q);
		Counter<Integer> c = Generics.newCounter();
		for (int i = 0; i < Q.size(); i++) {
			int w = Q.indexAt(i);
			IntegerArray vars = wordToVars.get(i);

			for (int v : vars) {
				if (w == v) {

				} else {
					c.incrementCount(v, 1);
				}
			}
		}

		SparseVector L = VectorUtils.toSparseVector(c);
		L.normalize();

		double mixture = 0.1;
		SparseVector ret = VectorMath.addAfterMultiply(Q, 1 - mixture, L, mixture);
		return ret;
	}

	public IntegerArrayMatrix mapWordToVariants(SparseVector Q) {
		IntegerArrayMatrix ret = new IntegerArrayMatrix(Q.size());
		for (int j = 0; j < Q.size(); j++) {
			int w = Q.indexAt(j);
			Integer lemma = wordToLemma.get(w);

			if (lemma == null) {
				continue;
			}

			IntegerArray tmp = new IntegerArray();

			Set<Integer> vars = lemmaToVars.get(lemma);

			if (vars != null) {
				tmp = new IntegerArray(vars);
			}

			ret.add(tmp);

		}
		return ret;
	}

}
