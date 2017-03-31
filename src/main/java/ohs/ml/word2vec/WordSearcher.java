package ohs.ml.word2vec;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Set;

import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class WordSearcher {

	public static void interact(WordSearcher searcher) throws Exception {

		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Enter word or sentence (EXIT to break): ");
				String line = br.readLine();
				if (line.toLowerCase().equals("exit")) {
					break;
				}

				List<String> words = StrUtils.split(StrUtils.normalizePunctuations(line));

				Counter<String> res = searcher.getSimilarWords(words, 20);

				System.out.println(res.toStringSortedByValues(true, true, 20, "\n"));
				System.out.println();
			}
		}
	}

	public static void interact2(WordSearcher searcher) throws Exception {

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

				Counter<String> res = searcher.getRelatedWords(words.get(0), words.get(1), words.get(2), 20);

				System.out.println(res.toStringSortedByValues(true, true, 20, "\n"));
				System.out.println();
			}
		}
	}

	private DenseVector q;

	private SparseVector sims;

	private DenseMatrix E;

	private Vocab vocab;

	private Set<String> stopwords;

	public WordSearcher(Vocab vocab, DenseMatrix E, Set<String> stopwords) {
		this.vocab = vocab;
		this.E = E;
		this.stopwords = stopwords;
		q = new DenseVector(E.colSize());
		sims = new SparseVector(ArrayUtils.range(vocab.size()));
	}

	private SparseVector computeSimilarity() {
		sims.setAll(0);

		for (int i = 0; i < E.size(); i++) {
			sims.setAt(i, VectorMath.cosine(q, E.row(i)));
		}

		return sims;
	}

	public Counter<String> getRelatedWords(String a, String b, String c, int top_k) {
		Counter<String> ret = Generics.newCounter();

		DenseVector wa = getVector(a);
		DenseVector wb = getVector(b);
		DenseVector wc = getVector(c);

		if (wa == null || wb == null || wc == null) {

		} else {
			q.setAll(0);

			for (int i = 0; i < wa.size(); i++) {
				q.addAt(i, wb.valueAt(i) - wa.valueAt(i) + wc.valueAt(i));
			}

			VectorMath.unitVector(q);

			computeSimilarity();

			ret = getSimilarWords(top_k);
		}

		return ret;
	}

	public Counter<String> getSimilarWords(Counter<String> wordCnts, int top_k) {
		int word_cnt = 0;

		q.setAll(0);

		for (String word : wordCnts.keySet()) {
			DenseVector v = getVector(word);
			if (v != null) {
				VectorMath.add(v, q, q);
				word_cnt++;
			}
		}

		Counter<String> ret = Generics.newCounter();

		if (word_cnt > 0) {
			VectorMath.multiply(q, 1f / word_cnt, q);

			computeSimilarity();

			ret = getSimilarWords(top_k);

		}

		return ret;
	}

	private Counter<String> getSimilarWords(int top_k) {
		Counter<String> ret = Generics.newCounter();
		sims.sortValues();

		for (int i = 0; i < top_k && i < E.size(); i++) {
			int w = sims.indexAt(i);
			double sim = sims.valueAt(i);
			ret.setCount(vocab.getObject(w), sim);
		}
		sims.sortIndexes();
		return ret;
	}

	public Counter<String> getSimilarWords(List<String> words, int top_k) {
		Counter<String> wordCnts = Generics.newCounter();
		for (String word : words) {
			word = word.toLowerCase();
			wordCnts.incrementCount(word, 1);
		}
		return getSimilarWords(wordCnts, top_k);
	}

	public Counter<String> getSimilarWords(String word, int top_k) {
		List<String> words = Generics.newArrayList();
		words.add(word);
		return getSimilarWords(words, top_k);
	}

	public DenseVector getVector(int w) {
		DenseVector ret = null;
		if (w >= 0 && w < E.rowSize()) {
			ret = E.row(w);
		}
		return ret;
	}

	public DenseVector getVector(String word) {
		return getVector(vocab.indexOf(word));
	}
}
