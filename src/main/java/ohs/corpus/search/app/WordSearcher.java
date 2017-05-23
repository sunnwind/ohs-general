package ohs.corpus.search.app;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class WordSearcher {

	public static void interact(WordSearcher ws) throws Exception {

		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Enter word or sentence (EXIT to break): ");
				String line = br.readLine();
				if (line.toLowerCase().equals("exit")) {
					break;
				}

				List<String> words = StrUtils.split(StrUtils.normalizePunctuations(line));

				Counter<String> res = ws.getSimilarWords(words, 20);

				System.out.println(res.toStringSortedByValues(true, true, 20, "\n"));
				System.out.println();
			}
		}
	}

	public static void interact2(WordSearcher ws) throws Exception {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Enter word or sentence (EXIT to break): ");
				String line = br.readLine();
				if (line.toLowerCase().equals("exit")) {
					break;
				}

				List<String> words = StrUtils.split(StrUtils.normalizePunctuations(line));

				if (words.size() < 3) {
					continue;
				}

				Counter<String> res = ws.getRelatedWords(words.get(0), words.get(1), words.get(2), 20);

				System.out.println(res.toStringSortedByValues(true, true, 20, "\n"));
				System.out.println();
			}
		}
	}

	public static void interact3(WordSearcher ws) throws Exception {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Enter word or sentence (EXIT to break): ");
				String line = br.readLine();
				if (line.toLowerCase().equals("exit")) {
					break;
				}

				List<String> words = StrUtils.split(line);

				if (words.size() != 2) {
					continue;
				}

				String word1 = words.get(0);
				String word2 = words.get(1);
				double cosine = ws.getCosine(word1, word2);

				System.out.printf("%s, %s, %f\n", word1, word2, cosine);
				System.out.println();
			}
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		String dir = MIRPath.WIKI_DIR;
		String emdFileName = dir + "glove_model_raf.ser";
		String vocabFileName = dir + "col/dc/vocab.ser.gz";

		Vocab vocab = new Vocab(vocabFileName);

		RandomAccessDenseMatrix ram = new RandomAccessDenseMatrix(emdFileName, true);

		WordSearcher ws = new WordSearcher(vocab, ram, null);
		WordSearcher.interact(ws);

		System.out.println("process ends.");
	}

	private DenseVector eq;

	private SparseVector cosines;

	private RandomAccessDenseMatrix E;

	private Vocab vocab;

	private Set<String> stopwords;

	private Map<Pair<Integer, Integer>, Double> cache;

	private boolean[] is_weighted;

	private boolean weight_embedding = false;

	public WordSearcher(Vocab vocab, RandomAccessDenseMatrix E, Set<String> stopwords) {
		this.vocab = vocab;
		this.E = E;
		this.stopwords = stopwords;
		eq = new DenseVector(E.colSize());
		cosines = new SparseVector(ArrayUtils.range(vocab.size()));
		cache = Generics.newWeakHashMap();
	}

	public double getCosine(int w1, int w2) throws Exception {
		double ret = 0;

		if (w1 >= 0 && w2 >= 0) {
			Double tmp = cache.get(Generics.newPair(w1, w2));

			if (tmp == null) {
				tmp = cache.get(Generics.newPair(w2, w1));
			}

			if (tmp == null) {
				tmp = VectorMath.dotProduct(E.row(w1), E.row(w2));
			}

			cache.put(Generics.newPair(w1, w2), tmp);

			ret = tmp.doubleValue();
		}
		return ret;
	}

	public double getCosine(String word1, String word2) throws Exception {
		int w1 = vocab.indexOf(word1);
		int w2 = vocab.indexOf(word2);
		return getCosine(w1, w2);
	}

	private SparseVector getCosines(DenseVector eq) throws Exception {
		cosines.setAll(0);
		cosines.sortIndexes();

		for (int w = 0; w < E.rowSize(); w++) {
			DenseVector ew = E.row(w);
			cosines.addAt(w, VectorMath.cosine(eq, ew));
		}
		cosines.sortValues();
		return cosines;
	}

	public RandomAccessDenseMatrix getEmbeddingMatrix() {
		return E;
	}

	public SparseVector getRelatedWords(int w1, int w2, int w3, int top_k) throws Exception {
		SparseVector ret = new SparseVector();

		DenseVector e1 = getVector(w1);
		DenseVector e2 = getVector(w2);
		DenseVector e3 = getVector(w3);

		Set<Integer> toRemove = Generics.newHashSet();
		toRemove.add(w1);
		toRemove.add(w2);
		toRemove.add(w3);

		if (e1 == null || e2 == null || e3 == null) {

		} else {
			eq.setAll(0);

			for (int i = 0; i < e1.size(); i++) {
				eq.addAt(i, e2.valueAt(i) - e1.valueAt(i) + e3.valueAt(i));
			}

			VectorMath.unitVector(eq);

			SparseVector cosines = getCosines(eq);

			ret = getSimilarWords(cosines, top_k, toRemove);

		}

		return ret;
	}

	public Counter<String> getRelatedWords(String word1, String word2, String word3, int top_k) throws Exception {
		SparseVector ret = getRelatedWords(vocab.indexOf(word1), vocab.indexOf(word2), vocab.indexOf(word3), top_k);
		return VectorUtils.toCounter(ret, vocab);
	}

	public SparseVector getSimilarWords(Collection<Integer> Q, int top_k) throws Exception {
		int word_cnt = 0;

		eq.setAll(0);

		Set<Integer> toRemove = Generics.newHashSet(Q.size());

		for (int w : Q) {
			DenseVector v = getVector(w);
			if (v != null) {
				VectorMath.add(v, eq);
				word_cnt++;
			}
			toRemove.add(w);
		}

		SparseVector ret = new SparseVector();

		if (word_cnt > 0) {
			VectorMath.multiply(eq, 1f / word_cnt, eq);

			cosines = getCosines(eq);

			ret = getSimilarWords(cosines, top_k, toRemove);

		}

		return ret;
	}

	public SparseVector getSimilarWords(int w, int top_k) throws Exception {
		List<Integer> words = Generics.newArrayList();
		words.add(w);
		return getSimilarWords(words, top_k);
	}

	public Counter<String> getSimilarWords(List<String> words, int top_k) throws Exception {
		List<Integer> q = Generics.newArrayList(words.size());

		for (String word : words) {
			if (stopwords != null && stopwords.contains(word)) {
				continue;
			}

			int w = vocab.indexOf(word);
			if (w == -1) {
				continue;
			}

			String word2 = vocab.getObject(w);

			q.add(w);
		}
		SparseVector ret = getSimilarWords(q, top_k);
		return VectorUtils.toCounter(ret, vocab);
	}

	private SparseVector getSimilarWords(SparseVector cosines, int top_k, Set<Integer> toRemove) {
		Counter<Integer> c = Generics.newCounter();
		for (int i = 0; i < cosines.size(); i++) {
			int w = cosines.indexAt(i);
			double cosine = cosines.valueAt(i);

			if (cosine > 1) {
				cosine = 1;
			} else if (cosine < -1) {
				cosine = -1;
			}

			if (toRemove.contains(w)) {
				continue;
			}

			if (c.size() == top_k) {
				break;
			}
			c.setCount(w, cosine);
		}

		SparseVector ret = VectorUtils.toSparseVector(c);
		ret.sortValues();
		return ret;
	}

	public Counter<String> getSimilarWords(String word, int top_k) throws Exception {
		List<String> words = Generics.newArrayList();
		words.add(word);
		return getSimilarWords(words, top_k);
	}

	public DenseVector getVector(int w) throws Exception {
		DenseVector ret = null;
		if (w >= 0 && w < E.rowSize()) {
			ret = E.row(w);
		}
		return ret;
	}

	public DenseVector getVector(String word) throws Exception {
		return getVector(vocab.indexOf(word));
	}

	public Vocab getVocab() {
		return vocab;
	}

	public void setWeightEmbedding(boolean weight_embedding) {
		this.weight_embedding = weight_embedding;
	}
}
