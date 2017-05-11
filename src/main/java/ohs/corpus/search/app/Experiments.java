package ohs.corpus.search.app;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Properties;

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
import ohs.io.TextFileReader;
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
import ohs.ir.medical.query.TrecCdsQuery;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.BidMap;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
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

		// e.runPhraseMapping();
		e.getDocPriors();
		// e.getDocPriors2();
		// e.runInitSearch();
		// e.runReranking1();

		// e.runFeedback();
		// e.runKLDFB();
		//
		// e.runKLDLemma();
		// e.runKLDFBLemma();

		// e.runKLDMetaMap();
		// e.runKLDFBMetaMap();

		// e.runFusion();
		// e.runVSM();

		// e.runKLDTune();
		// e.runKLDFBTune();

		// e.runKLDTmp();
		// e.runKLDFBTmp();

		// e.runBasicWithReraker();

		// e.runDependency();

		// e.runDependency2();
		// e.runDependencyEEM();

		// e.runBasicWithCommonness();
		// e.runBasicWithMeSH();
		// e.runThreeStageEEM();
		// e.runThreeStageEEMWithMeSH();
		// e.runWiki();
		// e.runTwoStageEEM();
		// e.runTwoStageEEMWithEmbedding();
		// e.runTwoStageEEMWithCommonness();
		// e.runTwoStageCBEEM();

		e.analyze1();
		// e.analyze2();

		System.out.println("process ends.");
	}

	private int top_k = 10000;

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

	public void analyze2() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		CounterMap<String, String> resData = FileUtils.readStringCounterMap(dataDir + "/res/sr_kld-fb.ser.gz");
		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);
		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		Performance p3 = new Performance(dataDir + "/res/p_kld-fb.ser.gz");

		// System.out.println(p3.toString(true));

		CounterMap<String, MetricType> queryScores = p3.getQueryScores().invert();
		CounterMap<String, MetricType> queryImprs = p3.getQueryImprovements().invert();

		int col_size = ds.getDocumentCollection().size();

		BidMap<String, Integer> idToSeq = Generics.newBidMap();

		if (FileUtils.exists(dataDir + "id_to_docseq.txt")) {
			TextFileReader reader = new TextFileReader(dataDir + "id_to_docseq.txt");
			while (reader.hasNext()) {
				String[] parts = reader.next().split("\t");
				idToSeq.put(parts[0], Integer.parseInt(parts[1]));
			}
			reader.close();
		} else {
			TextFileWriter writer = new TextFileWriter(dataDir + "id_to_docseq.txt");
			for (int i = 0; i < col_size; i++) {
				Pair<String, IntegerArray> p = ds.getDocumentCollection().get(i);
				String docid = p.getFirst();
				idToSeq.put(docid, i);

				writer.write(String.format("%s\t%d\n", docid, i));
			}
			writer.close();
		}

		TextFileWriter writer = new TextFileWriter(dataDir + "sr_match.txt");

		for (BaseQuery bq : bqs) {
			String qid = bq.getId();
			double p = queryScores.getCount(qid, MetricType.P);
			double ap = queryScores.getCount(qid, MetricType.AP);
			double ndcg = queryScores.getCount(qid, MetricType.NDCG);

			double impr_p = queryImprs.getCount(qid, MetricType.P);
			double impr_ap = queryImprs.getCount(qid, MetricType.AP);
			double impr_ndcg = queryImprs.getCount(qid, MetricType.NDCG);

			Counter<String> docScores = resData.getCounter(qid);
			Counter<String> docRels = relvData.getCounter(qid);

			List<String> docids = docScores.getSortedKeys();

			Map<Integer, Integer> rankToDoc = Generics.newHashMap();

			for (int i = 0; i < docids.size(); i++) {
				String docid = docids.get(i);
				double relevance = docRels.getCount(docid);
				int docseq = idToSeq.getValue(docid);

				if (relevance > 0 || i < 10) {
					rankToDoc.put(i + 1, docseq);
				}
			}

			if (p != 0) {
				continue;
			}

			List<Integer> ranks = Generics.newArrayList(rankToDoc.keySet());
			Collections.sort(ranks);

			SparseVector Q = ds.index(bq.getSearchText());

			System.out.printf("%s\t%d/%d\t%s\n", qid, rankToDoc.size(), docRels.size(), ranks);

			StringBuffer sb = new StringBuffer();
			sb.append("<query>");
			sb.append("\n" + bq.toString());

			for (int r : ranks) {
				int docseq = rankToDoc.get(r);
				double rel = docRels.getCount(idToSeq.getKey(docseq));
				IntegerArrayMatrix doc = ds.getDocumentCollection().getSents(docseq).getSecond();
				SparseVector dv = ds.getDocumentCollection().getDocVector(docseq);

				Counter<String> c = Generics.newCounter();

				for (int w : Q.indexes()) {
					String word = ds.getVocab().getObject(w);
					c.incrementCount(word, dv.prob(w));
				}

				sb.append("\n\n<doc>");
				sb.append(String.format("\nrank:\t%d", r));
				sb.append(String.format("\ndocseq:\t%s", docseq));
				sb.append(String.format("\nrelevance:\t%f", rel));
				sb.append(String.format("\nmatch:\t%f\n%s", c.totalCount(), c.toString(c.size())));

				Counter<String> cc = Generics.newCounter();

				for (int w : dv.indexes()) {
					if (ds.getWordFilter().filter(w)) {
						continue;
					}
					cc.incrementCount(ds.getVocab().getObject(w), dv.value(w));
				}

				sb.append("\n" + cc.toString(dv.size()));
				for (int i = 0; i < doc.size(); i++) {
					IntegerArray sent = doc.get(i);
					sb.append("\n");
					sb.append(StrUtils.join(" ", ds.getVocab().getObjects(sent)));
				}
				sb.append("\n</doc>");
			}
			sb.append("\n</query>");

			writer.write(sb.toString() + "\n\n");
		}

		writer.close();
	}

	public void analyze3() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		int col_size = ds.getDocumentCollection().size();

		Map<String, Integer> m = Generics.newHashMap();

		for (int i = 0; i < col_size; i++) {
			Pair<String, IntegerArray> p = ds.getDocumentCollection().get(i);
			String docid = p.getFirst();
			m.put(docid, i);
		}

		CounterMap<String, String> sr1 = FileUtils.readStringCounterMap(MIRPath.TREC_CDS_2014_DIR + "/res/sr_kld.ser.gz");
		CounterMap<String, String> sr2 = FileUtils.readStringCounterMap(MIRPath.TREC_CDS_2014_DIR + "/res/sr_kld-eem.ser.gz");
		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(MIRPath.TREC_CDS_2014_REL_JUDGE_FILE);

		Performance p1 = new Performance();
		Performance p2 = new Performance();
		p1.readObject(MIRPath.TREC_CDS_2014_DIR + "/res/perf_kld.ser.gz");
		p2.readObject(MIRPath.TREC_CDS_2014_DIR + "/res/perf_kld-eem.ser.gz");

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

			Counter<String> c1 = sr1.getCounter(qid);
			Counter<String> c2 = sr2.getCounter(qid);
			Counter<String> docRels = relvData.getCounter(qid);

			List<String> docids1 = c1.getSortedKeys();
			List<String> docids2 = c2.getSortedKeys();

			if (impr_p > 0 && (impr_ap < 0 || impr_ndcg < 0)) {
				System.out.println(qid);

				double ap2 = Metrics.averagePrecision(docids2, p1.getTopN(), docRels, true);
				double ndcg2 = Metrics.normalizedDiscountedCumulativeGain(docids2, p1.getTopN(), docRels);

				System.out.println();
			}
		}
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

		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());
		List<SparseVector> qData2 = Generics.newArrayList(bqs.size());
		List<SparseVector> dData2 = Generics.newArrayList(bqs.size());

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		LMScorer scorer = new LMScorer(ds);

		FeedbackBuilder fb = new FeedbackBuilder(ds.getVocab(), ds.getDocumentCollection(), ds.getInvertedIndex(), ds.getWordFilter());
		fb.setFbWordSize(10);

		for (int i = 0; i < qData1.rowSize(); i++) {
			Timer timer = Timer.newTimer();

			SparseVector Q = qData1.rowAt(i);
			SparseVector lm_q1 = VectorUtils.toSparseVector(Q);
			lm_q1.normalize();

			SparseVector scores = dData1.rowAt(i).subVector(top_k);
			SparseVector lm_fb = fb.buildRM1(scores, 0);
			SparseVector lm_q2 = fb.updateQueryModel(lm_q1, lm_fb);

			scores = scorer.score2(lm_q2, scores);
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

		List<String> phrss = Generics.newArrayList();

		{
			List<String> lines = FileUtils.readLinesFromText("../../data/medical_ir/phrs/phrs.txt");
			phrsCnts = Generics.newCounter(lines.size());

			for (String line : lines) {
				String[] ps = line.split("\t");
				String phrs = ps[0];
				int src_cnt = Integer.parseInt(ps[1]);
				if (phrs.split(" ").length > 1) {
					phrsCnts.setCount(phrs, src_cnt);
					phrss.add(phrs);
				}
			}
		}

		Collections.sort(phrss);

		List<SparseVector> qData = Generics.newArrayList(phrsCnts.size());
		List<SparseVector> dData = Generics.newArrayList(phrsCnts.size());

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		IntegerArray srcCnts = new IntegerArray(phrss.size());

		for (String phrs : phrss) {
			int src_cnt = (int) phrsCnts.getCount(phrs);
			SparseVector Q = ds.index(phrs);
			if (Q.size() > 0 && src_cnt > 2) {
				qData.add(Q);
				srcCnts.add(src_cnt);
			} else {
				// System.out.println(phrs);
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

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d, %d/%d, %s]", prog, i + 1, rs.length, r[1], ds.getDocumentCollection().size(),
						timer.stop());
			}
		}

		docPriors.writeObject(MIRPath.CLEF_EH_2017_DIR + "doc_phrs_prior.ser.gz");
		matchCnts.writeObject(MIRPath.CLEF_EH_2017_DIR + "doc_phrs_match.ser.gz");
	}

	// public void getDocPriors2() throws Exception {
	// Counter<String> phrsCnts = Generics.newCounter();
	//
	// {
	// List<String> lines = FileUtils.readLinesFromText("../../data/medical_ir/phrs/phrs.txt");
	// phrsCnts = Generics.newCounter(lines.size());
	//
	// for (String line : lines) {
	// String[] ps = line.split("\t");
	// String phrs = ps[0];
	// int src_cnt = Integer.parseInt(ps[1]);
	// if (phrs.split(" ").length > 1) {
	// phrsCnts.setCount(phrs, src_cnt);
	// }
	// }
	// }
	//
	// DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
	//
	// PhraseMapper<Integer> pm = new PhraseMapper<Integer>(PhraseMapper.createDict(phrsCnts.keySet(), ds.getVocab()));
	//
	// int[][] rs = BatchUtils.getBatchRanges(ds.getDocumentCollection().size(), 300);
	//
	// for (int i = 0; i < rs.length; i++) {
	// int[] r = rs[i];
	//
	// List<Pair<String, IntegerArray>> docs = ds.getDocumentCollection().getRange(r);
	//
	// for (int j = 0; j < docs.size(); j++) {
	// int dseq = j + r[0];
	// IntegerArray d = docs.get(j).getSecond();
	// List<Integer> words = Generics.newArrayList(d.size());
	//
	//
	// pm.map(words)
	// }
	// }
	//
	// docPriors.writeObject(MIRPath.CLEF_EH_2017_DIR + "doc_phrs_prior.ser.gz");
	// matchCnts.writeObject(MIRPath.CLEF_EH_2017_DIR + "doc_phrs_match.ser.gz");
	// }

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
		ds.setLemmaExpander(le);

		for (BaseQuery bq : bqs) {
			SparseVector Q = ds.index(bq.getSearchText());
			Q = le.expand(Q);
			qData.add(Q);
		}

		String modelName = "vsm";

		if (modelName.equals("mrf")) {
			ds.setScorer(new MRFScorer(ds));
		} else if (modelName.equals("wmrf")) {
			Counter<String> phrss = null;
			List<String> lines = FileUtils.readLinesFromText("../../data/medical_ir/phrs/phrs.txt");
			phrss = Generics.newCounter(lines.size());

			for (String line : lines) {
				String[] ps = line.split("\t");
				String phrs = ps[0];
				int source_cnt = Integer.parseInt(ps[1]);
				if (phrs.split(" ").length > 1) {
					phrss.setCount(phrs, source_cnt);
				}
			}

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
			List<SparseVector> res = ds.search(qData, null, 1);
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

	public void runReranking1() throws Exception {
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
