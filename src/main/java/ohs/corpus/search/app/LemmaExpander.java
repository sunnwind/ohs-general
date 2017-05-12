package ohs.corpus.search.app;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ohs.corpus.search.index.WordFilter;
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
import ohs.utils.StrUtils;

public class LemmaExpander {

	private WordFilter wf;

	private Vocab vocab;

	private Map<Integer, Integer> wordToLemma;

	private SetMap<Integer, Integer> lemmaToWords;

	public LemmaExpander(WordFilter wf, Map<String, String> m) throws Exception {
		this.wf = wf;
		this.vocab = wf.getVocab();
		wordToLemma = Generics.newHashMap(m.size());

		Set<Integer> L = Generics.newHashSet(m.size());

		for (Entry<String, String> e : m.entrySet()) {
			String word = e.getKey();
			String lemma = e.getValue();
			int w = vocab.indexOf(word);
			int l = vocab.indexOf(lemma);
			if (w >= 0 && l >= 0) {
				wordToLemma.put(w, l);
				L.add(l);
			}
		}

		lemmaToWords = Generics.newSetMap(L.size());

		for (Entry<Integer, Integer> e : wordToLemma.entrySet()) {
			int w = e.getKey();
			int l = e.getValue();
			lemmaToWords.put(l, w);
		}
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

	public void generate(SparseVector Q) {
		IntegerArrayMatrix varData = mapWordToVariants(Q);
		IntegerArrayMatrix allPaths = new IntegerArrayMatrix();

		generate(varData, 0, new IntegerArray(), allPaths);
	}

	public void generate(IntegerArrayMatrix varData, int i, IntegerArray path, IntegerArrayMatrix allPaths) {
		IntegerArray vars = varData.get(i);

		for (int j = 0; j < vars.size(); j++) {
			int w = vars.get(j);
			String word = vocab.getObject(w);

			IntegerArray newPath = new IntegerArray(path);
			newPath.add(w);

			if (i < varData.size() - 1) {
				generate(varData, i + 1, newPath, allPaths);
			} else {
				System.out.println(StrUtils.join(" ", vocab.getObjects(newPath)));
				allPaths.add(newPath);
			}
		}
	}

	public IntegerArrayMatrix mapWordToVariants(SparseVector Q) {
		IntegerArrayMatrix ret = new IntegerArrayMatrix(Q.size());
		for (int j = 0; j < Q.size(); j++) {
			int w = Q.indexAt(j);
			Integer lemma = wordToLemma.get(w);

			IntegerArray ws = new IntegerArray();

			if (lemma != null) {
				Set<Integer> vars = lemmaToWords.get(lemma);
				if (vars != null) {
					for (int v : vars) {
						String word = vocab.getObject(v);
						if (wf.filter(v) || word.contains("+") || word.contains("`")) {
							continue;
						}
						ws.add(v);
					}
				}
			}
			ret.add(ws);

		}
		return ret;
	}

}
