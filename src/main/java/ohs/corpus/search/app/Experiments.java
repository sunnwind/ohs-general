package ohs.corpus.search.app;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import org.tartarus.snowball.ext.PorterStemmer;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ohs.corpus.search.model.BM25Scorer;
import ohs.corpus.search.model.DocumentPriorEstimator;
import ohs.corpus.search.model.FeedbackBuilder;
import ohs.corpus.search.model.LMScorer;
import ohs.corpus.search.model.MRFScorer;
import ohs.corpus.search.model.VsmScorer;
import ohs.corpus.search.model.WeightedMRFScorer;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.DocumentIdMap;
import ohs.corpus.type.SimpleStringNormalizer;
import ohs.corpus.type.StringNormalizer;
import ohs.eden.keyphrase.mine.PhraseMapper;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.eval.MetricType;
import ohs.ir.eval.Metrics;
import ohs.ir.eval.Performance;
import ohs.ir.eval.PerformanceComparator;
import ohs.ir.eval.PerformanceEvaluator;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.medical.query.QueryReader;
import ohs.ir.medical.query.RelevanceReader;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class Experiments {

	public static int COR_TYPE = 4;

	public static int TASK_YEAR = 2015;

	public static String dataDir = MIRPath.TREC_CDS_2014_DIR;

	public static String resDir = dataDir + "res/";

	public static String idxDir = MIRPath.TREC_CDS_2014_COL_INDEX_DIR;

	public static String queryFileName = MIRPath.TREC_CDS_2014_QUERY_FILE;

	public static String relFileName = MIRPath.TREC_CDS_2014_REL_JUDGE_FILE;

	public static String stopwordFileName = MIRPath.STOPWORD_INQUERY_FILE;

	static {
		if (COR_TYPE == 0) {
			dataDir = MIRPath.OHSUMED_DIR;
			idxDir = MIRPath.OHSUMED_COL_DC_DIR;
			queryFileName = MIRPath.OHSUMED_QUERY_FILE;
			relFileName = MIRPath.OHSUMED_RELEVANCE_JUDGE_FILE;
		} else if (COR_TYPE == 1) {
			dataDir = MIRPath.TREC_CDS_2014_DIR;
			idxDir = MIRPath.TREC_CDS_2014_COL_DC_DIR;
			queryFileName = MIRPath.TREC_CDS_2014_QUERY_FILE;
			relFileName = MIRPath.TREC_CDS_2014_REL_JUDGE_FILE;
		} else if (COR_TYPE == 2) {
			dataDir = MIRPath.TREC_CDS_2015_DIR;
			idxDir = MIRPath.TREC_CDS_2014_COL_DC_DIR;
			queryFileName = MIRPath.TREC_CDS_2015_QUERY_A_FILE;
			relFileName = MIRPath.TREC_CDS_2015_REL_JUDGE_FILE;
		} else if (COR_TYPE == 3) {
			dataDir = MIRPath.TREC_CDS_2016_DIR;
			idxDir = MIRPath.TREC_CDS_2016_COL_DC_DIR;
			queryFileName = MIRPath.TREC_CDS_2016_QUERY_FILE;
			relFileName = MIRPath.TREC_CDS_2016_REL_JUDGE_FILE;
		} else if (COR_TYPE == 4) {
			dataDir = MIRPath.CLEF_EH_2017_DIR;
			idxDir = MIRPath.CLUEWEB_COL_DC_DIR;
			queryFileName = MIRPath.CLEF_EH_2016_QUERY_FILE;
			relFileName = MIRPath.CLEF_EH_2016_REL_JUDGE_FILE;
		}
		resDir = dataDir + "res/";
	}

	public static ListMap<String, Integer> getRelevantRanks(CounterMap<String, String> resData, CounterMap<String, String> relvData) {
		ListMap<String, Integer> ret = Generics.newListMap();
		for (String qid : resData.keySet()) {
			List<String> docids = resData.getCounter(qid).getSortedKeys();
			Counter<String> docRels = relvData.getCounter(qid);
			List<Integer> relRanks = Generics.newArrayList();

			for (int i = 0; i < docids.size(); i++) {
				String docid = docids.get(i);
				if (docRels.getCount(docid) > 0) {
					relRanks.add(i + 1);
				}
			}
			ret.put(qid, relRanks);
		}
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Experiments e = new Experiments();

		// e.getLemmas();

		// e.getRelevanceJudgeDocIds();

		// e.runPhraseMapping();
		// e.getDocPriors();
		// e.getDocPriors2();
		// e.runInitSearch();
//		e.runReranking();
		 e.runFeedback();

		// e.analyze1();
		// e.analyze2();

		System.out.println("process ends.");
	}

	private int top_k = 10000;

	public void getRelevanceJudgeDocIds() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		// DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		// ds.setTopK(top_k);

		DocumentIdMap dim = new DocumentIdMap(idxDir);

		Map<String, Integer> docIdToDocSeq = Generics.newHashMap();

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			String qid = bq.getId();
			Counter<String> relvs = relvData.getCounter(qid);
			List<String> dids = relvs.getSortedKeys();

			for (int j = 0; j < dids.size(); j++) {
				String did = dids.get(j);
				docIdToDocSeq.put(did, -1);
			}
		}

		for (int i = 0; i < dim.size(); i++) {
			String did = dim.get(i);

			if (docIdToDocSeq.containsKey(did)) {
				docIdToDocSeq.put(did, i);
			}
		}

		List<String> lines = Generics.newArrayList();

		for (String did : Generics.newTreeSet(docIdToDocSeq.keySet())) {
			int dseq = docIdToDocSeq.get(did);
			lines.add(String.format("%s\t%d", did, dseq));
		}

		FileUtils.writeStringCollectionAsText(dataDir + "relv_did_dseq.txt", lines);

	}

	public void analyze1() throws Exception {
		List<File> files = FileUtils.getFilesUnder(dataDir + "res/");

		Map<String, Performance> m = Generics.newHashMap();

		for (File file : files) {
			if (!file.getName().startsWith("p_")) {
				continue;
			}
			Performance p = new Performance();
			p.readObject(file.getPath());
			m.put(file.getName(), p);
		}

		String baseFileName = "p_lmd.ser.gz";

		Performance base = m.remove(baseFileName);

		for (Performance other : m.values()) {
			PerformanceComparator.compare(base, other);
		}

		StringBuffer sb = new StringBuffer();
		sb.append(String.format("FileName:\t%s", baseFileName));
		sb.append("\n" + base.toString(false));

		for (String fileName : m.keySet()) {
			Performance p = m.get(fileName);
			sb.append(String.format("\n\nFileName:\t%s", fileName));
			sb.append("\n" + p.toString(false));
		}

		FileUtils.writeAsText(dataDir + "analysis_perfs.txt", sb.toString());
	}

	private void collectSearchResults(DocumentSearcher ds, BaseQuery bq, SparseVector scores, CounterMap<String, String> resData, int top_k)
			throws Exception {

		scores = scores.subVector(top_k);
		scores.sortIndexes();

		for (int i = 0; i < scores.size() && i < top_k; i++) {
			int dseq = scores.indexAt(i);
			double score = scores.valueAt(i);
			String docid = ds.getDocumentCollection().get(dseq).getFirst();
			resData.incrementCount(bq.getId(), docid, score);
		}
	}

	private void collectSearchResults(DocumentIdMap dim, List<BaseQuery> bqs, List<SparseVector> dData, CounterMap<String, String> resData)
			throws Exception {

		Map<Integer, String> m = Generics.newHashMap();

		for (int i = 0; i < dData.size(); i++) {
			SparseVector scores = dData.get(i);
			scores = scores.subVector(top_k);
			for (int dseq : scores.indexes()) {
				m.put(dseq, "");
			}
			dData.set(i, scores);
		}

		for (int dseq : m.keySet()) {
			String did = dim.get(dseq);
			m.put(dseq, did);
		}

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector scores = dData.get(i);
			for (int j = 0; j < scores.size(); j++) {
				int dseq = scores.indexAt(j);
				double score = scores.valueAt(j);
				String docid = m.get(dseq);
				resData.incrementCount(bq.getId(), docid, score);
			}
		}
	}

	public void evaluate() throws Exception {
		CounterMap<String, String> sr1 = FileUtils.readStringCounterMap(MIRPath.TREC_CDS_2014_DIR + "sr_kld.ser.gz");
		CounterMap<String, String> sr2 = FileUtils.readStringCounterMap(MIRPath.TREC_CDS_2014_DIR + "sr_kld-fb.ser.gz");
		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(MIRPath.TREC_CDS_2014_REL_JUDGE_FILE);

		Performance p1 = new Performance();
		Performance p2 = new Performance();
		p1.readObject(MIRPath.TREC_CDS_2014_DIR + "perf_kld.ser.gz");
		p2.readObject(MIRPath.TREC_CDS_2014_DIR + "perf_kld-fb.ser.gz");

		PerformanceComparator.compare(p1, p2);

		System.out.println(p1.toString() + "\n");
		System.out.println(p2.toString() + "\n");

		CounterMap<String, MetricType> queryScores = p2.getQueryScores().invert();
		CounterMap<String, MetricType> queryImprs = p2.getQueryImprovements().invert();

		for (String qid : queryScores.keySet()) {
			double p = queryScores.getCount(qid, MetricType.P);
			double ap = queryScores.getCount(qid, MetricType.AP);
			double ndcg = queryScores.getCount(qid, MetricType.NDCG);

			double impr_p = queryImprs.getCount(qid, MetricType.P);
			double impr_ap = queryImprs.getCount(qid, MetricType.AP);
			double impr_ndcg = queryImprs.getCount(qid, MetricType.NDCG);

			if (impr_p > 0 && (impr_ap < 0 || impr_ndcg < 0)) {
				System.out.println(qid);

				Counter<String> c1 = sr1.getCounter(qid);
				Counter<String> c2 = sr2.getCounter(qid);
				Counter<String> docRels = relvData.getCounter(qid);

				List<String> docids1 = c1.getSortedKeys();
				List<String> docids2 = c2.getSortedKeys();

				double ap2 = Metrics.averagePrecision(docids2, p1.getTopN(), docRels, true);
				double ndcg2 = Metrics.normalizedDiscountedCumulativeGain(docids2, p1.getTopN(), docRels);

				System.out.println();
			}
		}
	}

	public void getLemmas() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		Vocab vocab = ds.getVocab();

		Properties prop = new Properties();
		prop.setProperty("annotators", "tokenize, ssplit, pos, lemma");

		StanfordCoreNLP coreNLP = new StanfordCoreNLP(prop);

		Map<String, String> m = Generics.newHashMap();

		for (int w = 0; w < vocab.size(); w++) {
			String word = vocab.getObject(w);
			int cnt = ds.getVocab().getCount(w);

			if (ds.getWordFilter().filter(w)) {
				continue;
			}

			if (ds.getVocab().getCount(w) < 50) {
				continue;
			}

			if (word.contains("=") || word.contains("&")) {
				continue;
			}

			Annotation anno = new Annotation(word);

			coreNLP.annotate(anno);

			List<CoreMap> sents = anno.get(SentencesAnnotation.class);

			for (CoreMap sentence : sents) {
				// traversing the words in the current sentence
				// a CoreLabel is a CoreMap with additional token-specific
				// methods

				for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
					// String word = token.get(TextAnnotation.class);
					String pos = token.get(PartOfSpeechAnnotation.class);
					String lemma = token.get(LemmaAnnotation.class);

					// if (!lemma.equals(word)) {
					m.put(word, lemma);
					// }
				}
			}
		}

		FileUtils.writeStringMapAsText(dataDir + "lemma.txt", m);
	}

	public void getStems() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		Vocab vocab = ds.getVocab();

		PorterStemmer stemmer = new PorterStemmer();

		Map<String, String> m = Generics.newHashMap();

		for (int w = 3; w < vocab.size(); w++) {
			if (ds.getWordFilter().filter(w)) {
				continue;
			}

			String word = vocab.get(w);
			stemmer.setCurrent(word);
			if (stemmer.stem()) {
				String stem = stemmer.getCurrent().trim();
				if (!word.equals(stem) && stem.length() > 0) {
					m.put(word, stem);
				}
			}
		}

		FileUtils.writeStringMapAsText(MIRPath.TREC_CDS_2014_DIR + "stems.txt", m);
	}

	public void runReranking() throws Exception {
		System.out.println(resDir);

		String modelName = "lmd";
		String fbName = "doc-prior";

		Performance p1 = new Performance(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		SparseMatrix qData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		System.out.println(p1.toString());

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		LMScorer scorer = new LMScorer(ds);

		// DenseVector docPriors = new DenseVector(dataDir + "doc_phrs_prior.ser.gz");
		// docPriors.normalize();

		// {
		// for (int i = 0; i < docPriors.size(); i++) {
		// double prior = docPriors.value(i);
		// prior = Math.exp(prior);
		// docPriors.set(i, prior);
		// }
		// docPriors.summation();
		// // DenseVector matchCnts = new DenseVector(dataDir + "doc_phrs_match.ser.gz");
		// }

		FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(), ds.getWordFilter());
		// fb.setFbWordSize(10);
		fb = null;

		DocumentPriorEstimator dpe = new DocumentPriorEstimator(ds.getDocumentCollection(), ds.getWordFilter());

		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());
		List<SparseVector> qData2 = Generics.newArrayList(bqs.size());
		List<SparseVector> dData2 = Generics.newArrayList(bqs.size());

		for (int i = 0; i < qData1.rowSize(); i++) {
			Timer timer = Timer.newTimer();

			SparseVector Q = qData1.rowAt(i);
			SparseVector lm_q1 = VectorUtils.toSparseVector(Q);
			lm_q1.normalize();

			SparseVector scores = dData1.rowAt(i).subVector(top_k);

			SparseVector lm_fb = new SparseVector(0);
			SparseVector lm_q2 = lm_q1;

			SparseVector docPriors = dpe.estimateDocLenPriors(scores.indexes());

			if (fb != null) {
				lm_fb = fb.buildRM1(scores, 0, docPriors);
				lm_q2 = fb.updateQueryModel(lm_q1, lm_fb);
			}

			for (int j = 0; j < scores.size(); j++) {
				int dseq = scores.indexAt(j);
				double score = scores.valueAt(j);
				double doc_prior = docPriors.valueAt(j);
				scores.setAt(j, score * doc_prior);
			}
			scores.summation();
			scores.sortValues();

			// scores = scorer.scoreDirect(lm_q2, scores);
			// scorer.postprocess(scores);

			dData2.add(scores);
			qData2.add(lm_q2);

			System.out.printf("No:\t%d\n", i);
			System.out.println(VectorUtils.toCounter(lm_q1, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_q2, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_fb, ds.getVocab()));
			System.out.println(timer.stop() + "\n");
		}

		// List<SparseVector> dData2 = ds.search(qData2, null, 2);

		collectSearchResults(new DocumentIdMap(idxDir), bqs, dData2, srData);

		String pFileName = resDir + String.format("%s_%s_%s.ser.gz", "p", modelName, fbName);
		String srFileName = resDir + String.format("%s_%s_%s.ser.gz", "sr", modelName, fbName);
		String qFileName = resDir + String.format("%s_%s_%s.ser.gz", "q", modelName, fbName);
		String dFileName = resDir + String.format("%s_%s_%s.ser.gz", "d", modelName, fbName);

		Performance p2 = PerformanceEvaluator.evalute(srData, relvData);
		p2.writeObject(pFileName);

		FileUtils.writeStringCounterMap(srFileName, srData);

		new SparseMatrix(qData2).writeObject(qFileName);
		new SparseMatrix(dData2).writeObject(dFileName);

		PerformanceComparator.compare(p1, p2);

		System.out.println(p2);
	}

	public void runFeedback() throws Exception {
		System.out.println(resDir);

		String modelName = "lmd";
		String fbName = "rm3";

		Performance p1 = new Performance(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		SparseMatrix qData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		System.out.println(p1.toString());

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		LMScorer scorer = new LMScorer(ds);

		DocumentPriorEstimator dpe = new DocumentPriorEstimator(ds.getDocumentCollection(), ds.getWordFilter());
		{

			// DenseVector docPriors = new DenseVector(dataDir + "doc_phrs_prior.ser.gz");
			// for (int i = 0; i < docPriors.size(); i++) {
			// double prior = docPriors.value(i);
			// prior = Math.exp(prior);
			// docPriors.set(i, prior);
			// }
			// docPriors.summation();
			// scorer.setDocPriors(docPriors);

			// DenseVector matchCnts = new DenseVector(dataDir + "doc_phrs_match.ser.gz");

		}

		FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(), ds.getWordFilter());
		// fb.setFbDocSize(10);
		// fb.setFbWordSize(10);

		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());
		List<SparseVector> qData2 = Generics.newArrayList(bqs.size());
		List<SparseVector> dData2 = Generics.newArrayList(bqs.size());

		for (int i = 0; i < qData1.rowSize(); i++) {
			Timer timer = Timer.newTimer();

			SparseVector Q = qData1.rowAt(i);
			SparseVector lm_q1 = VectorUtils.toSparseVector(Q);
			lm_q1.normalize();

			SparseVector scores = dData1.rowAt(i).subVector(top_k);

			SparseVector docPriors = dpe.estimateDocLenPriors(scores.indexes());

			SparseVector lm_fb = fb.buildRM1(scores, 0, docPriors);
			SparseVector lm_q2 = fb.updateQueryModel(lm_q1, lm_fb);

			scores = scorer.scoreDirect(lm_q2, scores);
			scorer.postprocess(scores);

			dData2.add(scores);
			qData2.add(lm_q2);

			System.out.printf("No:\t%d\n", i);
			System.out.println(VectorUtils.toCounter(lm_q1, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_q2, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_fb, ds.getVocab()));
			System.out.println(timer.stop() + "\n");
		}

		// List<SparseVector> dData2 = ds.search(qData2, null, 2);

		collectSearchResults(new DocumentIdMap(idxDir), bqs, dData2, srData);

		String pFileName = resDir + String.format("%s_%s_%s.ser.gz", "p", modelName, fbName);
		String srFileName = resDir + String.format("%s_%s_%s.ser.gz", "sr", modelName, fbName);
		String qFileName = resDir + String.format("%s_%s_%s.ser.gz", "q", modelName, fbName);
		String dFileName = resDir + String.format("%s_%s_%s.ser.gz", "d", modelName, fbName);

		Performance p2 = PerformanceEvaluator.evalute(srData, relvData);
		p2.writeObject(pFileName);

		FileUtils.writeStringCounterMap(srFileName, srData);

		new SparseMatrix(qData2).writeObject(qFileName);
		new SparseMatrix(dData2).writeObject(dFileName);

		PerformanceComparator.compare(p1, p2);

		System.out.println(p2);
	}

	public void runPhraseMapping() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);

		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		//
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2016_DIR + "emb/glove_ra.ser", true);

		WordSearcher ws = new WordSearcher(dc.getVocab(), E, null);

		StringNormalizer sn = new SimpleStringNormalizer(true);

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (BaseQuery bq : bqs) {
			String Q = sn.normalize(bq.getSearchText());
			List<String> words = StrUtils.split(Q);

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
			pm = new PhraseMapper<String>(PhraseMapper.createDict(phrss));
		}

		List<String> l1 = Generics.newArrayList();

		for (BaseQuery bq : bqs) {
			String Q = sn.normalize(bq.getSearchText());

			List<String> l2 = Generics.newArrayList();

			l2.add("QID:\t" + bq.getId());
			l2.add("Q:\t" + Q);

			List<String> words = StrUtils.split(Q);
			List<Pair<Integer, Integer>> ps = pm.map(words);

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

	public void getDocPriors() throws Exception {
		Timer timer = Timer.newTimer();

		Counter<String> phrsCnts = Generics.newCounter();

		{
			List<String> lines = FileUtils.readLinesFromText("../../data/medical_ir/phrs/phrs.txt");
			phrsCnts = Generics.newCounter(lines.size());

			for (String line : lines) {
				String[] ps = line.split("\t");
				String phrs = ps[0];
				int src_cnt = Integer.parseInt(ps[1]);

				Set<String> srcs = Generics.newHashSet();

				for (int i = 2; i < ps.length; i++) {
					String src = ps[i];
					srcs.add(src);
				}

				boolean is_in_mesh = false;
				boolean is_in_snomed = false;
				boolean is_in_cds = false;
				boolean is_in_wiki = false;
				boolean is_in_scopus = false;

				if (srcs.contains("mes")) {
					is_in_mesh = true;
				}

				if (srcs.contains("sno")) {
					is_in_snomed = true;
				}

				if (srcs.contains("cds")) {
					is_in_cds = true;
				}

				if (srcs.contains("wkt")) {
					is_in_wiki = true;
				}

				if (srcs.contains("sco")) {
					is_in_scopus = true;
				}

				if (is_in_mesh && is_in_snomed && is_in_wiki) {
					phrsCnts.setCount(phrs, src_cnt);
				}
			}
		}

		List<String> phrss = Generics.newArrayList(phrsCnts.keySet());

		Collections.sort(phrss);

		List<SparseVector> qData = Generics.newArrayList(phrsCnts.size());
		List<SparseVector> dData = Generics.newArrayList(phrsCnts.size());

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		for (String phrs : phrss) {
			SparseVector Q = ds.index(phrs);
			if (Q.size() > 0) {
				qData.add(Q);
			}
		}

		MRFScorer scorer = new MRFScorer(ds);
		scorer.setCliqueTypeMixtures(new double[] { 0, 1, 0 });
		ds.setScorer(scorer);

		DenseVector docPriors = new DenseVector(ds.getDocumentCollection().size());
		DenseVector matchCnts = new DenseVector(ds.getDocumentCollection().size());

		int window_size = 3;
		boolean keep_order = true;

		int thread_size = 3;

		int[][] rs = BatchUtils.getBatchRanges(qData.size(), thread_size);

		for (int i = 0; i < rs.length; i++) {
			int[] r = rs[i];

			List<SparseVector> qData2 = Generics.newArrayList();

			for (int j = r[0]; j < r[1]; j++) {
				qData2.add(qData.get(j));
			}

			List<SparseVector> dData2 = ds.search(qData2, null, thread_size);

			for (int j = 0; j < dData2.size(); j++) {
				SparseVector scores = dData2.get(j);

				for (int k = 0; k < scores.size(); k++) {
					int dseq = scores.indexAt(k);
					double score = scores.valueAt(k);

					docPriors.add(dseq, score);
					matchCnts.add(dseq, 1);
				}
			}

			int prog = BatchUtils.progress(i + 1, rs.length);
			//
			// if (prog > 0) {
			System.out.printf("[%d percent, %d/%d, %d/%d, %s]", prog, i + 1, rs.length, r[1], ds.getDocumentCollection().size(),
					timer.stop());
			// }
		}

		docPriors.writeObject(dataDir + "doc_phrs_prior.ser.gz");
		matchCnts.writeObject(dataDir + "doc_phrs_match.ser.gz");
	}

	public void getDocPriors2() throws Exception {
		Timer timer = Timer.newTimer();
		Counter<String> phrsCnts = Generics.newCounter();

		{
			List<String> lines = FileUtils.readLinesFromText("../../data/medical_ir/phrs/phrs.txt");
			phrsCnts = Generics.newCounter(lines.size());

			for (String line : lines) {
				String[] ps = line.split("\t");
				String phrs = ps[0];
				int src_cnt = Integer.parseInt(ps[1]);
				phrsCnts.setCount(phrs, src_cnt);
			}
		}

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		PhraseMapper<String> pm = new PhraseMapper<String>(PhraseMapper.createDict(phrsCnts.keySet()));

		int[][] rs = BatchUtils.getBatchRanges(ds.getDocumentCollection().size(), 300);

		Counter<String> phrsFreqs = Generics.newCounter(phrsCnts.size());

		if (FileUtils.exists("../../data/medical_ir/phrs/phrs_freq.txt")) {
			phrsFreqs = FileUtils.readStringCounterFromText("../../data/medical_ir/phrs/phrs_freq.txt");
		} else {
			for (String phrs : phrsCnts.keySet()) {
				phrsFreqs.setCount(phrs, 0);
			}

			for (int i = 0; i < rs.length; i++) {
				int[] r = rs[i];

				List<Pair<String, IntegerArray>> docs = ds.getDocumentCollection().getRange(r);

				for (int j = 0; j < docs.size(); j++) {
					int dseq = j + r[0];
					IntegerArray d = docs.get(j).getSecond();

					List<String> words = Generics.newArrayList(d.size());

					for (int w : d) {
						if (w != DocumentCollection.SENT_END) {
							words.add(ds.getVocab().getObject(w));
						}
					}

					List<Pair<Integer, Integer>> ps = pm.map(words);

					Counter<String> c = Generics.newCounter();

					for (Pair<Integer, Integer> p : ps) {
						String phrs = StrUtils.join(" ", words, p.getFirst(), p.getSecond());
						c.incrementCount(phrs, 1);
					}

					for (String phrs : c.keySet()) {
						phrsFreqs.incrementCount(phrs, 1);
					}
				}

				int prog = BatchUtils.progress(i + 1, rs.length);

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %d/%d, %s]\n", prog, i + 1, rs.length, r[1], ds.getDocumentCollection().size(),
							timer.stop());
				}
			}

			FileUtils.writeStringCounterAsText("../../data/medical_ir/phrs/phrs_freq.txt", phrsFreqs);
		}

		DenseVector docPriors = new DenseVector(ds.getDocumentCollection().size());

		for (int i = 0; i < rs.length; i++) {
			int[] r = rs[i];

			List<Pair<String, IntegerArray>> docs = ds.getDocumentCollection().getRange(r);

			for (int j = 0; j < docs.size(); j++) {
				int dseq = j + r[0];
				IntegerArray d = docs.get(j).getSecond();

				List<String> words = Generics.newArrayList(d.size());

				for (int w : d) {
					if (w != DocumentCollection.SENT_END) {
						words.add(ds.getVocab().getObject(w));
					}
				}

				List<Pair<Integer, Integer>> ps = pm.map(words);

				Counter<String> c = Generics.newCounter();

				for (Pair<Integer, Integer> p : ps) {
					String phrs = StrUtils.join(" ", words, p.getFirst(), p.getSecond());
					c.incrementCount(phrs, 1);
				}

				double sum = 0;

				for (String phrs : c.keySet()) {
					double cnt = c.getCount(phrs);
					double doc_freq = phrsFreqs.getCount(phrs);
					double tfidf = TermWeighting.tfidf(cnt, ds.getDocumentCollection().size(), doc_freq);
					sum += tfidf;
				}

				docPriors.add(dseq, sum);
			}

			int prog = BatchUtils.progress(i + 1, rs.length);

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d, %d/%d, %s]\n", prog, i + 1, rs.length, r[1], ds.getDocumentCollection().size(),
						timer.stop());
			}
		}

		docPriors.writeObject(dataDir + "doc_phrs_prior-tfidf.ser.gz");

	}

	public void runInitSearch() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);
		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());
		List<SparseVector> qData = Generics.newArrayList(bqs.size());
		List<SparseVector> dData = Generics.newArrayList(bqs.size());

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		ds.setTopK(top_k);

		LemmaExpander le = new LemmaExpander(ds.getWordFilter(),
				FileUtils.readStringHashMapFromText(MIRPath.CLEF_EH_2017_DIR + "lemma.txt"));

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector Q = ds.index(bq.getSearchText());
			// Q = le.expand(Q);

			qData.add(Q);
		}

		String modelName = "lmd";

		// {
		//
		// DenseVector docPriors = new DenseVector(dataDir + "doc_phrs_prior.ser.gz");
		//
		// for (int i = 0; i < docPriors.size(); i++) {
		// double prior = docPriors.value(i);
		// prior = Math.exp(prior);
		// docPriors.set(i, prior);
		// }
		// docPriors.summation();
		//
		// // DenseVector matchCnts = new DenseVector(dataDir + "doc_phrs_match.ser.gz");
		//
		// LMScorer scorer = (LMScorer) ds.getScorer();
		// }

		if (modelName.equals("mrf")) {
			ds.setScorer(new MRFScorer(ds));
		} else if (modelName.equals("wmrf")) {
			// Counter<String> phrsDocFreqs = FileUtils.readStringCounterFromText("../../data/medical_ir/phrs/phrs_freq.txt");
			List<String> phrss = FileUtils.readLinesFromText("../../data/medical_ir/phrs/phrs_medical.txt");

			ds.setScorer(new WeightedMRFScorer(ds, phrss));
		} else if (modelName.equals("bm25")) {
			ds.setScorer(new BM25Scorer(ds));
		} else if (modelName.equals("vsm")) {
			String normFileName = idxDir + "vsm_doc_norm.ser.gz";
			DenseVector docNorms = null;

			if (FileUtils.exists(normFileName)) {
				docNorms = new DenseVector(normFileName);
			} else {
				docNorms = VsmScorer.getDocNorms(ds.getDocumentCollection());
				docNorms.writeObject(normFileName);
			}
			ds.setScorer(new VsmScorer(ds, docNorms));
		}

		{
			List<SparseVector> res = ds.search(qData, null, 2);
			dData.addAll(res);
		}

		collectSearchResults(new DocumentIdMap(idxDir), bqs, dData, srData);

		// FileUtils.deleteFilesUnder(resDir);

		Performance p = PerformanceEvaluator.evalute(srData, relvData);

		p.writeObject(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		FileUtils.writeStringCounterMap(resDir + String.format("%s_%s.ser.gz", "sr", modelName), srData);
		new SparseMatrix(qData).writeObject(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		new SparseMatrix(dData).writeObject(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		System.out.println(p);
	}

	public void runRerankingEmbeddings() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);
		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		ds.setTopK(top_k);

		List<File> files = FileUtils.getFilesUnder(resDir);

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2016_DIR + "emb/glove_ra.ser", true);

		WordSearcher ws = new WordSearcher(new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR).getVocab(), E, null);

		for (File file : files) {
			if (!file.getName().startsWith("q_")) {
				continue;
			}

			SparseMatrix qData = new SparseMatrix(file.getPath());
			SparseMatrix dData = new SparseMatrix(file.getPath().replace("q_", "d_"));

			for (int i = 0; i < qData.size(); i++) {
				SparseVector Q = qData.get(i);
				SparseVector scores = dData.get(i);

				DenseVector eq = new DenseVector(E.colSize());
				int cnt = 0;

				for (int j = 0; j < Q.size(); j++) {
					int w = Q.indexAt(j);
					String word = ds.getVocab().getObject(w);
					DenseVector e = ws.getVector(word);

					if (e != null) {
						VectorMath.add(e, eq);
						cnt++;
					}
				}
				eq.multiply(1f / cnt);

				SparseVector cosines = new SparseVector(scores.size());

				for (int j = 0; j < scores.size(); j++) {
					int dseq = scores.indexAt(j);
					double score = scores.valueAt(j);
					SparseVector dv = ds.getDocumentCollection().getDocVector(dseq);
					DenseVector ed = new DenseVector(E.colSize());

					cnt = 0;
					for (int k = 0; k < dv.size(); k++) {
						int w = dv.indexAt(k);
						double cnt_w = dv.valueAt(k);
						String word = ds.getVocab().getObject(w);
						DenseVector e = ws.getVector(word);
						if (e != null) {
							VectorMath.addAfterMultiply(e, cnt_w, ed);
							cnt += cnt_w;
						}
					}
					ed.multiply(1f / cnt);

					double cosine = VectorMath.cosine(eq, ed);

					cosines.addAt(j, dseq, cosine);
				}
				cosines.sortValues();

				System.out.println(scores.toString());
				System.out.println(cosines.toString());
				System.out.println();
			}

		}

		// collectSearchResults(new DocumentIdMap(idxDir), bqs, dData, srData);
		//
		// // FileUtils.deleteFilesUnder(resDir);
		//
		// Performance p = PerformanceEvaluator.evalute(srData, relvData);
		//
		// p.writeObject(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		// FileUtils.writeStringCounterMap(resDir + String.format("%s_%s.ser.gz", "sr", modelName), srData);
		// new SparseMatrix(qData).writeObject(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		// new SparseMatrix(dData).writeObject(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		// System.out.println(p);
	}

}
