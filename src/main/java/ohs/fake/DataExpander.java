package ohs.fake;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.math.linear.SparseFieldVector;

import ohs.io.FileUtils;
import ohs.ir.search.model.ParsimoniousLanguageModelEstimator;
import ohs.math.ArrayMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LDocumentCollection;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class DataExpander {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Timer timer = Timer.newTimer();

		String newsFileName = FNPath.DATA_DIR + "news.ser";

		boolean read_news_file = false;

		LDocumentCollection C = new LDocumentCollection();

		if (read_news_file) {
			C = new LDocumentCollection();
			C.readObject(newsFileName);
		} else {
			C = new LDocumentCollection(10000);

			List<File> files = FileUtils.getFilesUnder(FNPath.NAVER_NEWS_COL_LINE_POS_DIR);

			Collections.reverse(files);

			for (int i = 0; i < files.size() && i < 1; i++) {
				File file = files.get(i);
				List<String> lines = FileUtils.readLinesFromText(file);

				for (String line : lines) {
					List<String> ps = StrUtils.split("\t", line);
					LDocument d = LDocument.newDocument(ps.get(4));
					d.getAttrMap().put("topic", ps.get(2));

					C.add(d);

					if (C.size() == 10000) {
						break;
					}
				}
			}

			// C.writeObejct(newsFileName);
		}

		DataExpander de = new DataExpander(C);
		de.expand1();

		System.out.printf("time:\t%s\n", timer.stop());

		System.out.println("process ends.");
	}

	private LDocumentCollection C;

	private Vocab vocab;

	public void expand1() {
		ListMap<String, Integer> lm = Generics.newListMap();

		for (int i = 0; i < C.size(); i++) {
			LDocument d = C.get(i);
			String topic = d.getAttrMap().get("topic");
			lm.put(topic, i);
		}

		for (String topic : lm.keySet()) {
			List<Integer> dseqs = lm.get(topic);

			for (int dseq : dseqs) {
				LDocument d = C.get(dseq);

				expand(d);

				System.out.println(d.toString());
				System.out.println();
			}
		}

		System.out.println();
	}

	public void expand(LDocument d) {
		ParsimoniousLanguageModelEstimator ple = new ParsimoniousLanguageModelEstimator(vocab);

		Counter<Integer> c = Generics.newCounter();
		for (String s : d.getTokenStrings(new int[] { 0, 1 }, "_")) {
			int w = vocab.indexOf(s);
			if (w < 0) {
				continue;
			}
			c.incrementCount(w, 1);
		}

		SparseVector sv1 = new SparseVector(c);
		SparseVector sv2 = ple.estimate(sv1);

		Counter<String> c1 = VectorUtils.toCounter(sv1, vocab);
		Counter<String> c2 = VectorUtils.toCounter(sv2, vocab);

		System.out.println(c1.toString());
		System.out.println(c2.toString());
		System.out.println();
	}

	public DataExpander(LDocumentCollection C) {
		this.C = C;

		createVocab();
	}

	private void createVocab() {
		vocab = new Vocab();

		Counter<Integer> wordCnts = Generics.newCounter();
		Counter<Integer> docFreqs = Generics.newCounter();

		for (LDocument d : C) {
			Counter<Integer> c = Generics.newCounter();
			for (String s : d.getTokenStrings(new int[] { 0, 1 }, "_")) {
				c.incrementCount(vocab.getIndex(s), 1);
			}

			wordCnts.incrementAll(c);
			docFreqs.incrementAll(c.keySet(), 1);
		}

		IntegerArray cnts = new IntegerArray(new int[vocab.size()]);
		IntegerArray doc_freqs = new IntegerArray(new int[vocab.size()]);

		for (int w : wordCnts.keySet()) {
			int cnt = (int) wordCnts.getCount(w);
			int doc_freq = (int) docFreqs.getCount(w);

			cnts.set(w, cnt);
			doc_freqs.set(w, doc_freq);
		}

		vocab.setDocFreqs(doc_freqs);
		vocab.setWordCnts(cnts);
		vocab.setDocCnt(C.size());

	}

	public void computeCentralities(LDocument d) {
		Indexer<String> idxer = Generics.newIndexer();

		CounterMap<Integer, Integer> cm = Generics.newCounterMap();
		CounterMap<String, String> cm2 = Generics.newCounterMap();
		Counter<Integer> c1 = Generics.newCounter();
		Counter<Integer> c2 = Generics.newCounter();
		int win_size = 5;

		for (LSentence s : d) {

			for (int i = 0; i < s.size(); i++) {
				LToken t1 = s.get(i);
				String w1 = t1.getString(0);
				String p1 = t1.getString(1);
				String l1 = w1 + "_" + p1;

				// if (!tfidfs.containsKey(w1)) {
				// continue;
				// }

				int idx1 = idxer.getIndex(l1);

				int size = Math.min(i + win_size, s.size());

				for (int j = i + 1; j < size; j++) {
					LToken t2 = s.get(j);
					String w2 = t2.getString(0);
					String p2 = t2.getString(1);
					String l2 = w2 + "_" + p2;

					// if (!tfidfs.containsKey(w2)) {
					// continue;
					// }

					int idx2 = idxer.getIndex(l2);
					double dist = j - i;
					double sim = 1d / dist;
					cm.incrementCount(idx1, idx2, sim);
					cm.incrementCount(idx2, idx1, sim);
				}

				cm2.incrementCount(p1, w1, 1);

				c1.incrementCount(idx1, 1);
				c2.incrementCount(idx1, 1d / (i + 1));
			}
		}

		DenseMatrix T = new SparseMatrix(cm).toDenseMatrix(idxer.size(), idxer.size());
		DenseVector C1 = new DenseVector(idxer.size());
		DenseVector C2 = new DenseVector(idxer.size());
		DenseVector B = new SparseVector(c2).toDenseVector(idxer.size());

		// {
		// for (int i = 0; i < idxer.size(); i++) {
		// String w1 = idxer.get(i).split("_")[0];
		// double tfidf1 = tfidfs.getCount(w1);
		//
		// for (int j = i; j < idxer.size(); j++) {
		// String w2 = idxer.get(j).split("_")[0];
		// double tfidf2 = tfidfs.getCount(w2);
		//
		// T.set(i, j, T.value(i, j) * tfidf1 * tfidf2);
		// T.set(j, i, T.value(i, j) * tfidf1 * tfidf2);
		// }
		// }
		//
		// T.multiply(1d / T.max());
		// }

		T.normalizeColumns();
		B.normalize();

		ArrayMath.randomWalk(T.values(), C1.values(), B.values(), 20, 5);
		ArrayMath.randomWalk(T.values(), C2.values(), null, 20, 5);

		CounterMap<String, String> cm3 = Generics.newCounterMap();

		for (Entry<String, Double> e : VectorUtils.toCounter(C1, idxer).entrySet()) {
			String l = e.getKey();
			String w = l.split("_")[0];
			String p = l.split("_")[1];
			cm3.incrementCount(p, w, e.getValue());
		}

		System.out.println(cm3.toString());

		System.out.println(VectorUtils.toCounter(c1, idxer));
		System.out.println(VectorUtils.toCounter(B, idxer));
		System.out.println(VectorUtils.toCounter(C1, idxer));
		System.out.println(VectorUtils.toCounter(C2, idxer));
		System.out.println();
	}

}
