package ohs.fake;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map;

import javax.rmi.CORBA.Stub;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.DocumentIdMap;
import ohs.corpus.type.EnglishTokenizer;
import ohs.corpus.type.StringTokenizer;
import ohs.eden.keyphrase.mine.PhraseMapper;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.io.TextFileWriter;
import ohs.ir.eval.MetricType;
import ohs.ir.eval.Metrics;
import ohs.ir.eval.Performance;
import ohs.ir.eval.PerformanceComparator;
import ohs.ir.eval.PerformanceEvaluator;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.medical.query.QueryReader;
import ohs.ir.medical.query.RelevanceReader;
import ohs.ir.search.app.DocumentSearcher;
import ohs.ir.search.app.WordSearcher;
import ohs.ir.search.model.BM25Scorer;
import ohs.ir.search.model.FeedbackBuilder;
import ohs.ir.search.model.LanguageModelScorer;
import ohs.ir.search.model.MarkovRandomFieldsScorer;
import ohs.ir.search.model.ParsimoniousLanguageModelEstimator;
import ohs.ir.search.model.VSMScorer;
import ohs.ir.search.model.WeightedMRFScorer;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class NewsSearchExperiment {

	public static String dataDir = MIRPath.TREC_PM_2017_DIR;

	public static String idxDir = FNPath.NAVER_NEWS_COL_DC_DIR;

	public static String queryFileName = MIRPath.TREC_PM_2017_QUERY_FILE;

	public static String relFileName = "";

	public static String resDir = dataDir + "res/medline/";

	public static String stopwordFileName = MIRPath.STOPWORD_INQUERY_FILE;

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		NewsSearchExperiment e = new NewsSearchExperiment();

		int[] corTypes = { 0, 1, 2 };

		// for (int i = 0; i < corTypes.length; i++) {
		// int corType = corTypes[i];
		//
		// if (corType != 1) {
		// continue;
		// }
		//
		// // if (corType < 1) {
		// // continue;
		// // }
		dataDir = MIRPath.TREC_CDS_2016_DIR;
		idxDir = FNPath.NAVER_NEWS_COL_DC_DIR;
		queryFileName = MIRPath.TREC_CDS_2016_QUERY_FILE;
		relFileName = MIRPath.TREC_CDS_2016_REL_JUDGE_FILE;
		//
		// e.runInitSearch();
		// // e.runReranking();
		// // e.runReranking2();
		// // e.runReranking3();
		// // e.runReranking4();
		// // e.formatTrecOutput();
		// // e.test();
		// }

		e.test1();

		// e.runInitSearch();
		// e.runReranking();
		// e.runReranking2();
		// e.runReranking3();
		// e.test();

		// e.analyze1();
		// e.analyze2();

		// e.formatTrecOutput();

		System.out.println("process ends.");
	}

	private int top_k = 100;

	public void buildMedicalFeedbackQueries() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		List<SparseVector> qData = Generics.newArrayList(bqs.size());

		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2016_COL_DC_DIR, stopwordFileName);
		ds.setTopK(top_k);

		Vocab vocab2 = DocumentCollection.readVocab(idxDir + "vocab.ser");

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector Q = ds.index(bq.getSearchText());
			qData.add(Q);
		}

		FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(), ds.getWordFilter());

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector Q = qData.get(i);
			SparseVector lm_q = Q.copy();
			lm_q.normalize();

			SparseVector scores = ds.search(lm_q);
			SparseVector lm_fb = fb.buildRelevanceModel1(scores);
			SparseVector lm_q2 = fb.updateQueryModel(lm_q, lm_fb);

			lm_q2 = VectorUtils.toSparseVector(scores, ds.getVocab(), vocab2);
			lm_q2.normalize();

			qData.set(i, lm_q2);

			System.out.println(bq.toString());
			System.out.println(VectorUtils.toCounter(lm_fb, ds.getVocab()));
			System.out.println();
		}

		String modelName = "lmd-fb-cds";
		new SparseMatrix(qData).writeObject(resDir + String.format("%s_%s.ser.gz", "q", modelName));
	}

	public void test1() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, null);
		ds.setTopK(top_k);
		ds.setStringNormalizer(new StringTokenizer());
		ds.getWordFilter().buildStopIds(null);

		String[] fileNames = { "Mission1_sample_1차수정본.txt", "Mission2_sample_1차수정본.txt" };

		ParsimoniousLanguageModelEstimator ple = new ParsimoniousLanguageModelEstimator(ds.getDocumentCollection());
		// ple.setDocumentMixture(0.5);
		Vocab vocab = ds.getVocab();

		for (int u = 0; u < fileNames.length && u < 1; u++) {
			File inFile = new File(FNPath.DATA_DIR + "샘플데이터_1차수정본", fileNames[u]);

			List<String> bqs = FileUtils.readLinesFromText(inFile);

			for (int i = 0; i < bqs.size() && i < 10; i++) {
				String bq = bqs.get(i);

				List<String> ps = StrUtils.split("\t", bq);
				ps = StrUtils.unwrap(ps);

				String id = ps.get(0);
				String label = ps.get(1);

				File outFile = new File(FNPath.DATA_DIR + "rel_new/", String.format("%s.txt", id));

				MDocument d = MDocument.newDocument(ps.get(2));

				Counter<Integer> c1 = Generics.newCounter();
				Counter<Integer> c2 = Generics.newCounter();
				Counter<Integer> c3 = Generics.newCounter();

				for (int j = 0; j < d.size(); j++) {
					MSentence s = d.get(j);
					String st = StrUtils.join(" ", s.getTokenStrings(0));
					SparseVector Q = ds.index(st);

					for (int k = 0; k < Q.size(); k++) {
						int w = Q.indexAt(k);
						double cnt = Q.valueAt(k);
						c1.incrementCount(w, cnt);

						if (j == 0) {
							c2.incrementCount(w, cnt);
						} else {
							c3.incrementCount(w, cnt);
						}
					}
				}

				double score = 0;

				for (int w : c2.keySet()) {
					if (c3.containsKey(w)) {
						score++;
					}
				}

				score = 1d * score / c2.size();

				SparseVector Q = new SparseVector(c1);

				SparseVector lm = ple.estimate(Q);

				System.out.println(StrUtils.join(" ", d.get(0).getTokenStrings(0)));
				System.out.printf("dot score: %f\n", score);
				System.out.printf("label: %s\n", label);
				System.out.println(VectorUtils.toCounter(Q, vocab));
				System.out.println(VectorUtils.toCounter(lm, vocab));
				System.out.println();
			}
		}

		ds.close();
	}

	public void runInitSearch() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, null);
		ds.setTopK(top_k);
		ds.setStringNormalizer(new StringTokenizer());
		ds.getWordFilter().buildStopIds(null);

		System.out.println(ds.getDocumentCollection().toString());

		String modelName = "lmd";

		if (modelName.equals("mrf")) {
			ds.setScorer(new MarkovRandomFieldsScorer(ds));
			MarkovRandomFieldsScorer scorer = (MarkovRandomFieldsScorer) ds.getScorer();
			// scorer.setDocumentPriors(qualityPriors);
			scorer.setPhraseSize(2);
		} else if (modelName.equals("bm25")) {
			ds.setScorer(new BM25Scorer(ds));
		} else if (modelName.equals("vsm")) {
			String normFileName = idxDir + "vsm_doc_norm.ser.gz";
			DenseVector docNorms = null;
			if (FileUtils.exists(normFileName)) {
				docNorms = new DenseVector(normFileName);
			} else {
				docNorms = VSMScorer.getDocNorms(ds.getDocumentCollection());
				docNorms.writeObject(normFileName);
			}
			ds.setScorer(new VSMScorer(ds, docNorms));
		}

		// {
		// List<SparseVector> res = ds.search(qData, null, 2);
		// dData.addAll(res);
		// }

		DocumentCollection dc = ds.getDocumentCollection();

		FileUtils.deleteFilesUnder(FNPath.DATA_DIR + "rel_new/");

		DecimalFormat df = new DecimalFormat("0000");

		String[] fileNames = { "Mission1_sample_1차수정본.txt", "Mission2_sample_1차수정본.txt" };

		for (int u = 0; u < fileNames.length; u++) {
			File inFile = new File(FNPath.DATA_DIR + "샘플데이터_1차수정본", fileNames[u]);

			List<String> bqs = FileUtils.readLinesFromText(inFile);

			for (int i = 0; i < bqs.size(); i++) {
				String bq = bqs.get(i);

				List<String> ps = StrUtils.split("\t", bq);
				ps = StrUtils.unwrap(ps);

				String id = ps.get(0);

				File outFile = new File(FNPath.DATA_DIR + "rel_new/", String.format("%s.txt", id));

				MDocument d = MDocument.newDocument(ps.get(2));

				String searchText = StrUtils.join(" ", d.get(0).getTokenStrings(0));

				SparseVector Q = ds.index(searchText);
				SparseVector docs = ds.search(Q);

				// SparseVector docs = dData.get(i);

				StringBuffer sb = new StringBuffer();
				sb.append(String.format("Q: %s", StrUtils.join(" ", d.get(0).getTokenStrings(0))));

				for (int j = 0; j < docs.size(); j++) {
					int dseq = docs.indexAt(j);
					double score = docs.valueAt(j);
					Pair<String, String> p = dc.getText(dseq);

					sb.append("\n\n");
					sb.append(String.format("\nno:\t%d", j + 1));
					sb.append(String.format("\ndseq:\t%d", dseq));
					sb.append(String.format("\nscore:\t%f", score));
					sb.append(String.format("\ntext:\n%s", p.getSecond()));
				}

				System.out.println(searchText);
				// writer.write(sb.toString());

				// TextFileWriter writer = new TextFileWriter(FNPath.DATA_DIR + "rel_news.txt");

				FileUtils.writeAsText(outFile.getPath(), sb.toString());
			}
		}
	}

	public void runPhraseMapping() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);

		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		//
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2016_DIR + "emb/glove_ra.ser", true);

		WordSearcher ws = new WordSearcher(dc.getVocab(), E, null);

		StringTokenizer st = new EnglishTokenizer();

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (BaseQuery bq : bqs) {
			List<String> words = st.tokenize(bq.getSearchText());

			for (String word : words) {
				if (!cm.containsKey(word)) {
					Counter<String> c = ws.getSimilarWords(word, 5);
					cm.setCounter(word, c);
				}
			}
		}

		// WordSearcher.interact(ws);

		PhraseMapper<String> pm = null;

		{
			List<String> lines = FileUtils.readLinesFromText("../../data/medical_ir/phrs.txt");
			List<String> phrss = Generics.newArrayList(lines.size());
			for (String line : lines) {
				String[] ps = line.split("\t");
				String phrs = ps[0];
				if (phrs.split(" ").length > 1) {
					phrss.add(phrs);
				}
			}
			pm = new PhraseMapper<String>(PhraseMapper.createTrie(phrss));
		}

		List<String> l1 = Generics.newArrayList();

		for (BaseQuery bq : bqs) {
			List<String> words = st.tokenize(bq.getSearchText());

			List<String> l2 = Generics.newArrayList();

			l2.add("QID:\t" + bq.getId());
			l2.add("Q:\t" + words);

			List<IntPair> ps = pm.map(words);

			if (ps.size() > 0) {
				for (Pair<Integer, Integer> p : ps) {
					int s = p.getFirst();
					int e = p.getSecond();

					StringBuffer sb = new StringBuffer();
					sb.append(String.format("[%d-%d,", s, e));

					for (int j = s; j < e; j++) {
						String w = words.get(j);
						sb.append(" ");
						sb.append(w);
					}

					l2.add(String.format("P\t%d-%d\t%s", s, e, StrUtils.join(" ", words, s, e)));
				}
			}

			for (int i = 0; i < words.size(); i++) {
				String word = words.get(i);
				Counter<String> c = cm.getCounter(word);
				l2.add(String.format("W\t%d\t%s\t", i, word, c.toString()));
			}

			l1.add(StrUtils.join("\n", l2));
		}

		FileUtils.writeAsText(MIRPath.CLEF_EH_2017_DIR + "qs.txt", StrUtils.join("\n\n", l1));
	}

	public void runReranking() throws Exception {
		System.out.println(resDir);

		String modelName = "lmd";
		String fbName = "rm3";

		SparseMatrix qData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		ds.getWordFilter().buildStopIds2();

		LanguageModelScorer scorer = (LanguageModelScorer) ds.getScorer();

		FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(), ds.getWordFilter());

		for (int i = 0; i < qData.rowSize(); i++) {
			Timer timer = Timer.newTimer();

			SparseVector Q = qData.rowAt(i);
			SparseVector lm_q1 = VectorUtils.toSparseVector(Q);
			lm_q1.normalize();

			SparseVector scores = dData.rowAt(i);

			SparseVector lm_fb = fb.buildRelevanceModel1(scores, 0, null);
			SparseVector lm_q2 = fb.updateQueryModel(lm_q1, lm_fb);

			SparseVector scores2 = scorer.scoreFromCollection(lm_q2, scores);
			scorer.postprocess(scores2);

			dData.set(i, scores2);
			qData.set(i, lm_q2);

			System.out.printf("No:\t%d\n", i);
			System.out.println(VectorUtils.toCounter(lm_q1, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_q2, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_fb, ds.getVocab()));
			System.out.println(timer.stop() + "\n");
		}

		// CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());

		// collectSearchResults(new DocumentIdMap(idxDir), bqs, dData, srData);

		String pFileName = resDir + String.format("%s_%s_%s.ser.gz", "p", modelName, fbName);
		String srFileName = resDir + String.format("%s_%s_%s.ser.gz", "sr", modelName, fbName);
		String qFileName = resDir + String.format("%s_%s_%s.ser.gz", "q", modelName, fbName);
		String dFileName = resDir + String.format("%s_%s_%s.ser.gz", "d", modelName, fbName);

		// Performance p2 = PerformanceEvaluator.evalute(srData, relvData);
		// PerformanceComparator.compare(p1, p2);
		// p2.writeObject(pFileName);

		// FileUtils.writeStringCounterMap(srFileName, srData);

		new SparseMatrix(qData).writeObject(qFileName);
		new SparseMatrix(dData).writeObject(dFileName);

	}

	public void runReranking2() throws Exception {
		System.out.println(resDir);

		String modelName = "lmd";
		String fbName = "fb-rw";

		Performance p1 = new Performance(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		SparseMatrix qData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		ds.setUseCache(false);
		ds.getWordFilter().buildStopIds2();

		LanguageModelScorer scorer = (LanguageModelScorer) ds.getScorer();

		FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(), ds.getWordFilter());

		double[] mixtures = { 4, 3, 3 };
		ArrayMath.normalize(mixtures);

		for (int i = 0; i < qData.rowSize(); i++) {
			Timer timer = Timer.newTimer();

			SparseVector Q = qData.rowAt(i);
			SparseVector lm_q1 = VectorUtils.toSparseVector(Q);
			lm_q1.normalize();

			SparseVector scores = dData.rowAt(i);

			SparseVector lm_fb1 = fb.buildRelevanceModel1(scores, 0, null);
			SparseVector lm_fb2 = fb.buildRandomWalkModel(Q, scores);

			SparseVector lm_q2 = VectorMath.addAfterMultiply(new SparseVector[] { lm_q1, lm_fb1, lm_fb2 }, mixtures);

			SparseVector scores2 = scorer.scoreFromCollection(lm_q2, scores);
			scorer.postprocess(scores2);

			dData.set(i, scores2);
			qData.set(i, lm_q2);

			System.out.printf("No:\t%d\n", i);
			System.out.println(VectorUtils.toCounter(lm_q1, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_q2, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_fb1, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_fb2, ds.getVocab()));
			System.out.println(timer.stop() + "\n");
		}

		String pFileName = resDir + String.format("%s_%s_%s.ser.gz", "p", modelName, fbName);
		String srFileName = resDir + String.format("%s_%s_%s.ser.gz", "sr", modelName, fbName);
		String qFileName = resDir + String.format("%s_%s_%s.ser.gz", "q", modelName, fbName);
		String dFileName = resDir + String.format("%s_%s_%s.ser.gz", "d", modelName, fbName);

		new SparseMatrix(qData).writeObject(qFileName);
		new SparseMatrix(dData).writeObject(dFileName);
	}

	public void runReranking3() throws Exception {
		System.out.println(resDir);

		String idxDir2 = "";
		String resDir2 = "";

		{
			String mType = "medline";
			String cType = "clinical";

			if (idxDir.contains(mType)) {
				idxDir2 = idxDir.replace(mType, cType);
				resDir2 = resDir.replace(mType, cType);
			} else if (idxDir.contains(cType)) {
				idxDir2 = idxDir.replace(cType, mType);
				resDir2 = resDir.replace(cType, mType);
			}
		}

		String modelName = "lmd";
		String fbName = "fb-eem";

		SparseMatrix qData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		SparseMatrix qData2 = new SparseMatrix(resDir2 + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData2 = new SparseMatrix(resDir2 + String.format("%s_%s.ser.gz", "d", modelName));

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds1 = new DocumentSearcher(idxDir, stopwordFileName);
		ds1.setUseCache(false);
		ds1.getWordFilter().buildStopIds2();

		DocumentSearcher ds2 = new DocumentSearcher(idxDir2, stopwordFileName);
		ds2.getWordFilter().buildStopIds2();

		LanguageModelScorer scorer = (LanguageModelScorer) ds1.getScorer();

		FeedbackBuilder fb1 = new FeedbackBuilder(ds1.getDocumentCollection(), ds1.getInvertedIndex(),
				ds1.getWordFilter());
		FeedbackBuilder fb2 = new FeedbackBuilder(ds2.getDocumentCollection(), ds2.getInvertedIndex(),
				ds2.getWordFilter());

		List<FeedbackBuilder> fbs = Generics.newArrayList();
		fbs.add(fb1);
		fbs.add(fb2);

		for (int i = 0; i < qData1.rowSize(); i++) {
			Timer timer = Timer.newTimer();

			SparseVector Q = qData1.rowAt(i);
			SparseVector lm_q1 = VectorUtils.toSparseVector(Q);
			lm_q1.normalize();

			SparseVector scores1 = dData1.rowAt(i);
			SparseVector scores2 = dData2.rowAt(i);

			List<SparseVector> scoreData = Generics.newArrayList();
			scoreData.add(scores1);
			scoreData.add(scores2);

			SparseVector lm_fb = FeedbackBuilder.buildExternalExpansionModel(fbs, scoreData, null, null);

			SparseVector lm_q2 = fb1.updateQueryModel(lm_q1, lm_fb);

			SparseVector newScores1 = scorer.scoreFromCollection(lm_q2, scores1);
			scorer.postprocess(newScores1);

			dData1.set(i, newScores1);
			qData1.set(i, lm_q2);

			System.out.printf("No:\t%d\n", i);
			System.out.println(VectorUtils.toCounter(lm_q1, ds1.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_q2, ds1.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_fb, ds1.getVocab()));
			System.out.println(timer.stop() + "\n");
		}

		String pFileName = resDir + String.format("%s_%s_%s.ser.gz", "p", modelName, fbName);
		String srFileName = resDir + String.format("%s_%s_%s.ser.gz", "sr", modelName, fbName);
		String qFileName = resDir + String.format("%s_%s_%s.ser.gz", "q", modelName, fbName);
		String dFileName = resDir + String.format("%s_%s_%s.ser.gz", "d", modelName, fbName);

		new SparseMatrix(qData1).writeObject(qFileName);
		new SparseMatrix(dData1).writeObject(dFileName);
	}

	public void runReranking4() throws Exception {
		System.out.println(resDir);

		String idxDir2 = "";
		String resDir2 = "";

		{
			String mType = "medline";
			String cType = "clinical";

			if (idxDir.contains(mType)) {
				idxDir2 = idxDir.replace(mType, cType);
				resDir2 = resDir.replace(mType, cType);
			} else if (idxDir.contains(cType)) {
				idxDir2 = idxDir.replace(cType, mType);
				resDir2 = resDir.replace(cType, mType);
			}
		}

		String modelName = "lmd";
		String fbName = "fb-rw-eem-mix";

		SparseMatrix qData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		SparseMatrix qData2 = new SparseMatrix(resDir2 + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData2 = new SparseMatrix(resDir2 + String.format("%s_%s.ser.gz", "d", modelName));

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds1 = new DocumentSearcher(idxDir, stopwordFileName);
		ds1.setUseCache(false);
		ds1.getWordFilter().buildStopIds2();

		DocumentSearcher ds2 = new DocumentSearcher(idxDir2, stopwordFileName);
		ds2.getWordFilter().buildStopIds2();

		LanguageModelScorer scorer = (LanguageModelScorer) ds1.getScorer();

		FeedbackBuilder fb1 = new FeedbackBuilder(ds1.getDocumentCollection(), ds1.getInvertedIndex(),
				ds1.getWordFilter());
		FeedbackBuilder fb2 = new FeedbackBuilder(ds2.getDocumentCollection(), ds2.getInvertedIndex(),
				ds2.getWordFilter());

		List<FeedbackBuilder> fbs = Generics.newArrayList();
		fbs.add(fb1);
		fbs.add(fb2);

		double[] mixtures = { 4, 3, 3 };

		for (int i = 0; i < qData1.rowSize(); i++) {
			Timer timer = Timer.newTimer();

			SparseVector Q = qData1.rowAt(i);
			SparseVector lm_q1 = VectorUtils.toSparseVector(Q);
			lm_q1.normalize();

			SparseVector scores1 = dData1.rowAt(i);
			SparseVector scores2 = dData2.rowAt(i);

			List<SparseVector> scoreData = Generics.newArrayList();
			scoreData.add(scores1);
			scoreData.add(scores2);

			SparseVector lm_fb1 = FeedbackBuilder.buildExternalExpansionModel(fbs, scoreData, null, null);
			SparseVector lm_fb2 = FeedbackBuilder.buildCorpusMixtureRandomWalkModel(Q, fbs, scoreData);

			SparseVector lm_q2 = VectorMath.addAfterMultiply(new SparseVector[] { lm_q1, lm_fb1, lm_fb2 }, mixtures);

			SparseVector newScores1 = scorer.scoreFromCollection(lm_q2, scores1);
			scorer.postprocess(newScores1);

			dData1.set(i, newScores1);
			qData1.set(i, lm_q2);

			System.out.printf("No:\t%d\n", i);
			System.out.println(VectorUtils.toCounter(lm_q1, ds1.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_q2, ds1.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_fb2, ds1.getVocab()));
			System.out.println(timer.stop() + "\n");
		}

		String pFileName = resDir + String.format("%s_%s_%s.ser.gz", "p", modelName, fbName);
		String srFileName = resDir + String.format("%s_%s_%s.ser.gz", "sr", modelName, fbName);
		String qFileName = resDir + String.format("%s_%s_%s.ser.gz", "q", modelName, fbName);
		String dFileName = resDir + String.format("%s_%s_%s.ser.gz", "d", modelName, fbName);

		new SparseMatrix(qData1).writeObject(qFileName);
		new SparseMatrix(dData1).writeObject(dFileName);
	}

}
