package ohs.corpus.search.app;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
import ohs.corpus.search.model.FeedbackBuilder;
import ohs.corpus.search.model.LMScorer;
import ohs.corpus.search.model.MRFScorer;
import ohs.corpus.search.model.TranslationModelScorer;
import ohs.corpus.search.model.VsmScorer;
import ohs.corpus.type.SimpleStringNormalizer;
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
import ohs.ir.medical.general.NLPUtils;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.medical.query.QueryReader;
import ohs.ir.medical.query.RelevanceReader;
import ohs.ir.medical.query.TrecCdsQuery;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.CommonMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.NeuralNet;
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
		// e.getStems();
		// e.getLemmas();
		// e.runInitSearch();
		e.runFeedback();
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

		// e.analyze1();
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

		String baseFileName = "p_kld.ser.gz";

		Performance base = m.remove(baseFileName);

		for (Performance other : m.values()) {
			PerformanceComparator.compare(base, other);
		}

		StringBuffer sb = new StringBuffer();
		sb.append(String.format("FileName:\t%s", baseFileName));
		sb.append("\n" + base.toString(true));

		for (String fileName : m.keySet()) {
			Performance p = m.get(fileName);
			sb.append(String.format("\n\nFileName:\t%s", fileName));
			sb.append("\n" + p.toString(true));
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

	public void buildBigrams() throws Exception {
		String idxDir = MIRPath.TREC_CDS_2014_COL_INDEX_DIR;

		DocumentSearcher searcher = new DocumentSearcher(idxDir, MIRPath.STOPWORD_INQUERY_FILE);

		Vocab vocab = searcher.getVocab();

		CounterMap<String, String> cm = FileUtils.readStringCounterMapFromText(MIRPath.WIKI_DIR + "medical_word_word.txt");
		CounterMap<Integer, Integer> cm2 = Generics.newCounterMap();

		for (String word1 : cm.keySet()) {
			int w1 = vocab.indexOf(word1);
			if (w1 >= 0) {
				Counter<String> c = cm.getCounter(word1);
				for (String word2 : c.keySet()) {
					double cnt = c.getCount(word2);
					int w2 = vocab.indexOf(word2);
					if (w2 >= 0) {
						cm2.incrementCount(w1, w2, cnt);
					}
				}
			}
		}

		SparseMatrix sm = VectorUtils.toSparseMatrix(cm2);
		sm.writeObject(MIRPath.WIKI_DIR + "medical_bigram.ser.gz");
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

	private void collectSearchResults(DocumentSearcher ds, List<BaseQuery> bqs, List<SparseVector> scoreData,
			CounterMap<String, String> resData, int top_k) throws Exception {

		Map<Integer, String> m = Generics.newHashMap();

		for (int i = 0; i < scoreData.size(); i++) {
			SparseVector scores = scoreData.get(i);
			scores = scores.subVector(top_k);
			for (int dseq : scores.indexes()) {
				m.put(dseq, "");
			}
			scoreData.set(i, scores);
		}

		for (int dseq : m.keySet()) {
			String did = ds.getDocumentCollection().get(dseq).getFirst();
			m.put(dseq, did);
		}

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector scores = scoreData.get(i);
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

		for (int w = 3; w < vocab.size(); w++) {
			if (ds.getWordFilter().filter(w)) {
				continue;
			}
			String word = vocab.get(w);

			Annotation anno = new Annotation(word);

			coreNLP.annotate(anno);

			List<CoreMap> sents = anno.get(SentencesAnnotation.class);

			for (CoreMap sentence : sents) {
				// traversing the words in the current sentence
				// a CoreLabel is a CoreMap with additional token-specific
				// methods

				StringBuffer sb = new StringBuffer();

				for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
					// String word = token.get(TextAnnotation.class);
					String pos = token.get(PartOfSpeechAnnotation.class);
					String lemma = token.get(LemmaAnnotation.class);

					if (!lemma.equals(word)) {
						m.put(word, lemma);
					}
				}
			}
		}

		// for(String lemma : cm.keySet()){
		// cm.setCount(lemma, lemma, 1);
		// }

		FileUtils.writeStringMapAsText(MIRPath.TREC_CDS_2014_DIR + "lemmas.txt", m);
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

	public void runBasicWithReraker() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		Vocab vocab = ds.getVocab();

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(MIRPath.TREC_CDS_2014_QUERY_FILE);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(MIRPath.TREC_CDS_2014_REL_JUDGE_FILE);

		CounterMap<String, String> resData1 = Generics.newCounterMap();
		CounterMap<String, String> resData2 = Generics.newCounterMap();

		int top_k = 1000;

		List<SparseVector> qs1 = Generics.newArrayList();
		List<SparseVector> qs2 = Generics.newArrayList();

		NeuralNet nn = new NeuralNet();
		nn.readObject(MIRPath.TREC_CDS_2014_DIR + "reranker.ser.gz");
		nn.setIsTesting(true);

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser");

		FeatureExtractor featExt = new FeatureExtractor(ds, E);

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(bq);

			SparseVector Q1 = ds.index(bq.getSearchText());
			SparseVector lm_q1 = ds.getQueryModel(Q1);

			SparseVector docScores1 = ds.search(lm_q1);

			docScores1.keepTopN(top_k);
			docScores1.sortValues();

			for (int j = 0; j < docScores1.size(); j++) {
				int docseq = docScores1.indexAt(j);
				double score = docScores1.valueAt(j);

				DenseVector feats = new DenseVector(featExt.extract(Q1, docseq).values());
				DenseVector rscores = nn.score(feats);
				double rscore = 1 - rscores.value(0);
				score = score * rscore;
				docScores1.setAt(j, score);
			}

			docScores1.summation();
			docScores1.sortValues();

			// SparseVector lm_q2 = ds.getQueryModel(lm_q1, true);

			// System.out.printf("lm_q1\t%s", VectorUtils.toCounter(lm_q1,
			// vocab));
			// System.out.printf("\nlm_q2\t%s\n\n", VectorUtils.toCounter(lm_q2,
			// vocab));

			// SparseVector docScores2 = ds.search(lm_q2);

			qs1.add(lm_q1);
			// qs2.add(lm_q2);

			collectSearchResults(ds, bq, docScores1, resData1, top_k);
			// collectSearchResults(ds, bq, docScores2, resData2, top_k);
		}

		Performance p1 = PerformanceEvaluator.evalute(resData1, relvData);
		// Performance p2 = PerformanceEvaluator.evalute(resData2, relvData);

		// p1.writeObject(MIRPath.TREC_CDS_2014_DIR + "/res/perf_kld.ser.gz");
		// p2.writeObject(MIRPath.TREC_CDS_2014_DIR +
		// "/res/perf_kld-fb.ser.gz");
		//
		// FileUtils.writeStringCounterMap(MIRPath.TREC_CDS_2014_DIR +
		// "/res/sr_kld.ser.gz", resData1);
		// FileUtils.writeStringCounterMap(MIRPath.TREC_CDS_2014_DIR +
		// "/res/sr_kld-fb.ser.gz", resData2);
		//
		// new SparseMatrix(qs1).writeObject(MIRPath.TREC_CDS_2014_DIR +
		// "/res/q_kld.ser.gz");
		// new SparseMatrix(qs2).writeObject(MIRPath.TREC_CDS_2014_DIR +
		// "/res/q_kld-fb.ser.gz");

		System.out.println(p1);
		System.out.println();
		// System.out.println(p2);
	}

	public void runDependency() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> resData = Generics.newCounterMap();

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser", true);
		WordSearcher raws = new WordSearcher(ds.getVocab(), E, null);

		TranslationModelScorer depScorer = new TranslationModelScorer(ds.getVocab(), ds.getDocumentCollection(), ds.getInvertedIndex(),
				raws, ds.getWordFilter());

		SparseMatrix qData1 = new SparseMatrix(dataDir + "res/q_kld-lemma.ser.gz");
		SparseMatrix dData1 = new SparseMatrix(dataDir + "res/d_kld-lemma.ser.gz");

		List<SparseVector> qData2 = Generics.newArrayList();
		List<SparseVector> dData2 = Generics.newArrayList();

		int top_n = 2000;

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(tcq.toString());

			SparseVector lm_q1 = qData1.rowAt(i);
			SparseVector scores1 = dData1.rowAt(i);

			scores1 = scores1.subVector(top_n);
			scores1 = depScorer.score(lm_q1, scores1);

			SparseVector lm_fb = ds.getFeedbackBuilder().buildRM1(scores1, 0);
			SparseVector lm_q2 = ds.updateQueryModel(lm_q1, lm_fb);

			SparseVector scores2 = ds.search(lm_q2);
			scores2 = scores2.subVector(top_k);

			collectSearchResults(ds, bq, scores2, resData, top_k);

			qData2.add(lm_q2);
			dData2.add(scores2);
		}

		Performance p = PerformanceEvaluator.evalute(resData, relvData);
		p.writeObject(dataDir + "res/p_kld-fb-dep.ser.gz");

		FileUtils.writeStringCounterMap(dataDir + "res/sr_kld-fb-dep.ser.gz", resData);

		new SparseMatrix(qData2).writeObject(dataDir + "res/q_kld-fb-dep.ser.gz");
		new SparseMatrix(dData2).writeObject(dataDir + "res/d_kld-fb-dep.ser.gz");

		System.out.println(p);
		System.out.println();
	}

	public void runDependency2() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		Vocab vocab = ds.getVocab();

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> resData = Generics.newCounterMap();

		RandomAccessDenseMatrix E1 = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser", true);
		WordSearcher raws = new WordSearcher(ds.getVocab(), E1, null);

		SparseMatrix E2 = new SparseMatrix(MIRPath.TREC_CDS_2014_DIR + "cosine_word.ser.gz");
		E2 = VectorUtils.symmetric(E2);

		// CounterMap<String, String> cm = VectorUtils.toCounterMap(M, vocab,
		// vocab);
		//
		// System.out.println(cm);
		// System.out.println(cm.getCounter("cancer"));
		// System.out.println();

		TranslationModelScorer depScorer = new TranslationModelScorer(vocab, ds.getDocumentCollection(), ds.getInvertedIndex(), raws,
				ds.getWordFilter());
		// depScorer.setMixtureTrsmSem(0.5);
		depScorer.setSematicSims(E2);

		SparseMatrix qData1 = new SparseMatrix(dataDir + "res/q_kld.ser.gz");
		SparseMatrix dData1 = new SparseMatrix(dataDir + "res/d_kld.ser.gz");

		List<SparseVector> qData2 = Generics.newArrayList();
		List<SparseVector> dData2 = Generics.newArrayList();

		int top_n = 1000;

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(tcq.toString());

			SparseVector lm_q1 = qData1.rowAt(i);
			SparseVector scores1 = dData1.rowAt(i);

			// SparseVector lm_q2 = VectorUtils.toSparseVector(c);
			// lm_q2.normalize();

			// System.out.printf("lm_q1:\t%s\n", VectorUtils.toCounter(lm_q1,
			// vocab));
			// System.out.printf("lm_q2:\t%s\n", VectorUtils.toCounter(lm_q2,
			// vocab));

			scores1 = scores1.subVector(top_n);
			scores1 = depScorer.score(lm_q1, scores1);

			SparseVector lm_fb = ds.getFeedbackBuilder().buildTRM1(scores1, 0);
			SparseVector lm_q2 = ds.updateQueryModel(lm_q1, lm_fb);

			SparseVector scores2 = ds.search(lm_q2);
			scores2 = scores2.subVector(top_k);

			collectSearchResults(ds, bq, scores2, resData, top_k);

			qData2.add(lm_q2);
			dData2.add(scores2);
		}

		Performance p = PerformanceEvaluator.evalute(resData, relvData);
		p.writeObject(dataDir + "res/p_kld-fb-dep-2.ser.gz");

		FileUtils.writeStringCounterMap(dataDir + "res/sr_kld-fb-dep-2.ser.gz", resData);

		new SparseMatrix(qData2).writeObject(dataDir + "res/q_kld-fb-dep-2.ser.gz");
		new SparseMatrix(dData2).writeObject(dataDir + "res/d_kld-fb-dep-2.ser.gz");

		System.out.println(p);
		System.out.println();
	}

	public void runDependencyEEM() throws Exception {
		DocumentSearcher ds1 = new DocumentSearcher(idxDir, stopwordFileName);
		DocumentSearcher ds2 = new DocumentSearcher(MIRPath.WIKI_COL_INDEX_DIR, stopwordFileName);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> resData = Generics.newCounterMap();

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser");

		int top_n = 500;

		TranslationModelScorer depScorer1 = new TranslationModelScorer(ds1.getVocab(), ds1.getDocumentCollection(), ds1.getInvertedIndex(),
				null, ds1.getWordFilter());
		TranslationModelScorer depScorer2 = new TranslationModelScorer(ds2.getVocab(), ds2.getDocumentCollection(), ds2.getInvertedIndex(),
				null, ds2.getWordFilter());

		SparseMatrix qData1 = new SparseMatrix();
		SparseMatrix dData1 = new SparseMatrix();

		qData1.readObject(dataDir + "res/q_kld.ser.gz");
		dData1.readObject(dataDir + "res/d_kld.ser.gz");

		boolean[] flags1 = { true };
		boolean[] flags2 = { true };
		boolean[] flags3 = { true };

		for (Boolean f1 : flags1) {
			for (Boolean f2 : flags2) {
				for (Boolean f3 : flags3) {
					List<SparseVector> qData2 = Generics.newArrayList();
					List<SparseVector> dData2 = Generics.newArrayList();

					for (int i = 0; i < bqs.size(); i++) {
						BaseQuery bq = bqs.get(i);
						TrecCdsQuery tcq = (TrecCdsQuery) bq;

						System.out.println(tcq.toString());

						SparseVector lm_q1 = qData1.rowAt(i);
						SparseVector lm_q2 = VectorUtils.toSparseVector(lm_q1, ds1.getVocab(), ds2.getVocab());

						SparseVector scores1 = dData1.rowAt(i);
						SparseVector scores2 = ds2.search(lm_q2);

						SparseVector tmpScores1 = scores1.subVector(top_n);
						SparseVector tmpScores2 = scores2.subVector(top_n);

						tmpScores1 = depScorer1.score(lm_q1, tmpScores1);
						tmpScores2 = depScorer2.score(lm_q2, tmpScores2);

						if (f1) {
							SparseVector lm_fb1 = ds1.getFeedbackBuilder().buildRM1(tmpScores1);
							lm_q1 = ds1.updateQueryModel(lm_q1, lm_fb1);
							scores1 = ds1.search(lm_q1);
							tmpScores1 = scores1.subVector(top_n);
						}

						if (f2) {
							SparseVector lm_fb2 = ds2.getFeedbackBuilder().buildRM1(tmpScores2);
							lm_q2 = ds2.updateQueryModel(lm_q2, lm_fb2);
							scores2 = ds2.search(lm_q2);
							tmpScores2 = scores2.subVector(top_n);
						}

						List<DocumentSearcher> dss = Generics.newArrayList();
						dss.add(ds1);
						dss.add(ds2);

						List<SparseVector> scoreData = Generics.newArrayList();
						scoreData.add(tmpScores1);
						scoreData.add(tmpScores2);

						SparseVector lm_fb = ds1.getFeedbackBuilder().buildEEM(dss, scoreData);
						lm_q1 = ds1.updateQueryModel(lm_q1, lm_fb);

						if (f3) {
							lm_q1 = ds1.getQueryModel(lm_q1);
						}

						scores1 = ds1.search(lm_q1);
						scores1 = scores1.subVector(top_k);

						collectSearchResults(ds1, bq, scores1, resData, top_k);

						qData2.add(lm_q1);
						dData2.add(scores1);
					}

					StringBuffer sb = new StringBuffer();
					for (Boolean f : new Boolean[] { f1, f2, f3 }) {
						sb.append(f.toString().charAt(0) + "");
					}

					String outFileName1 = String.format(dataDir + "/res/p_kld-fb-dep-eem-%s.ser.gz", sb.toString());
					String outFileName2 = String.format(dataDir + "/res/sr_kld-fb-dep-eem%s.ser.gz", sb.toString());
					String outFileName3 = String.format(dataDir + "/res/q_kld-fb-dep-eem-%s.ser.gz", sb.toString());
					String outFileName4 = String.format(dataDir + "/res/d_kld-fb-dep-eem-%s.ser.gz", sb.toString());

					Performance p = PerformanceEvaluator.evalute(resData, relvData);
					p.writeObject(outFileName1);

					FileUtils.writeStringCounterMap(outFileName2, resData);

					new SparseMatrix(qData2).writeObject(outFileName3);
					new SparseMatrix(dData2).writeObject(outFileName4);

					System.out.println(p);
					System.out.println();

				}
			}
		}
	}

	public void runFeedback() throws Exception {
		System.out.println(resDir);

		String modelName = "mrf";
		String fbName = "rm3";

		Performance p1 = new Performance(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		SparseMatrix qData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData1 = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());
		List<SparseVector> qData2 = Generics.newArrayList(bqs.size());
		List<SparseVector> dData2 = Generics.newArrayList(bqs.size());

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		// ds.getFeedbackBuilder().setUseDocumentPrior(true);
		// ds.getFeedbackBuilder().setFbWordSize(15);

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);

			System.out.println(bq);

			SparseVector Q = qData1.rowAt(i);
			SparseVector lm_q1 = VectorUtils.toSparseVector(Q);
			lm_q1.normalize();

			SparseVector scores1 = dData1.rowAt(i);
			scores1 = scores1.subVector(top_k);

			SparseVector lm_fb = ds.getFeedbackBuilder().buildRM1(scores1, 0);

			// Counter<String> c = VectorUtils.toCounter(lm_fb, ds.getVocab());
			// System.out.println(c.toString());

			// for (int j = 0; j < scores1.size(); j++) {
			// int dseq = scores1.indexAt(j);
			// Pair<String, String> p = ds.getDocumentCollection().getText(dseq);
			//
			// SparseVector dv = ds.getDocumentCollection().getDocVector(dseq);
			//
			// Counter<String> c2 = Generics.newCounter();
			//
			// for (int w : Q.indexes()) {
			// c2.setCount(ds.getVocab().getObject(w), dv.value(w));
			// }
			//
			// System.out.println(p.getFirst());
			// System.out.println(p.getSecond());
			// System.out.println(c2.toString());
			// System.out.println();
			// }

			SparseVector lm_q2 = ds.updateQueryModel(lm_q1, lm_fb);

			scores1.sortIndexes();
			SparseVector scores2 = ds.search(lm_q2, scores1);
			scores2 = scores2.subVector(top_k);

			qData2.add(lm_q2);
			dData2.add(scores2);

			collectSearchResults(ds, bq, scores2, srData, top_k);
		}

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

	public void runFusion() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> srData = Generics.newCounterMap();

		List<SparseVector> qData = Generics.newArrayList();
		List<SparseVector> dData = Generics.newArrayList();

		DenseVector docNorms = null;

		String docNormFileName = idxDir + "doc_norm.ser.gz";

		if (FileUtils.exists(docNormFileName)) {
			docNorms = new DenseVector(docNormFileName);
		} else {
			docNorms = VsmScorer.getDocNorms(ds.getVocab(), ds.getDocumentCollection());
			docNorms.writeObject(docNormFileName);
		}

		LMScorer scorer1 = (LMScorer) ds.getScorer();
		BM25Scorer scorer2 = new BM25Scorer(ds);
		VsmScorer scorer3 = new VsmScorer(ds);

		// ds.setScorer(new BM25Scorer(ds.getVocab(), ds.getDocumentStore(),
		// ds.getInvertedIndex()));

		LemmaExpander le = new LemmaExpander(ds.getVocab(), FileUtils.readStringHashMapFromText(MIRPath.TREC_CDS_2014_DIR + "lemmas.txt"));

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(bq);

			SparseVector Q = ds.index(bq.getSearchText());

			// Q = le.expand(Q);

			ds.setScorer(scorer1);
			SparseVector scores1 = ds.search(Q);

			ds.setScorer(scorer2);
			SparseVector scores2 = ds.search(Q, scores1);

			ds.setScorer(scorer3);
			SparseVector scores3 = ds.search(Q, scores1);

			scores1.sortIndexes();
			scores2.sortIndexes();
			scores3.sortIndexes();

			// scores1.normalize();
			// scores2.normalize();
			// scores3.normalize();

			Counter<Integer> c = Generics.newCounter(scores1.size());

			DenseVector weights = new DenseVector(new double[] { 60, 30, 10 });
			weights.normalizeAfterSummation();

			for (int j = 0; j < scores1.size(); j++) {
				int docseq = scores1.indexAt(j);
				double v1 = scores1.valueAt(j);
				double v2 = scores2.value(docseq);
				double v3 = scores3.value(docseq);

				DenseVector vals = new DenseVector(new double[] { v1, v2, v3 });

				// double v4 = v1 * v2 * v3;
				double v4 = VectorMath.dotProduct(weights, vals);

				if (v4 > 0) {
					c.setCount(docseq, v4);
				}
			}

			SparseVector scores4 = new SparseVector(c);
			scores4.sortValues();

			scores4 = scores4.subVector(top_k);

			qData.add(Q);
			dData.add(scores4);

			collectSearchResults(ds, bq, scores4, srData, top_k);
		}

		Performance p = PerformanceEvaluator.evalute(srData, relvData);
		p.writeObject(dataDir + "res/p_fs.ser.gz");

		FileUtils.writeStringCounterMap(dataDir + "res/sr_fs.ser.gz", srData);

		new SparseMatrix(qData).writeObject(dataDir + "res/q_fs.ser.gz");
		new SparseMatrix(dData).writeObject(dataDir + "res/d_fs.ser.gz");

		System.out.println(p);
	}

	public void runInitSearch() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);
		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());
		List<SparseVector> qData = Generics.newArrayList(bqs.size());
		List<SparseVector> dData = Generics.newArrayList(bqs.size());

		CounterMap<String, Integer> cm = Generics.newCounterMap();

		for (String qid : relvData.keySet()) {
			Counter<String> rels = relvData.getCounter(qid);

			for (Entry<String, Double> e : rels.entrySet()) {
				cm.incrementCount(qid, e.getValue().intValue(), 1);
			}
		}

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		ds.setTopK(top_k);
		ds.setUseFeedback(false);

		String modelName = "mrf";

		if (modelName.equals("mrf")) {
			ds.setScorer(new MRFScorer(ds));
		}

		PhraseMapper<Integer> pm = null;

		{
			List<String> lines = FileUtils.readLinesFromText("../../data/medical_ir/phrs.txt");
			List<String> phrss = Generics.newArrayList(lines.size());
			for (String line : lines) {
				String[] ps = line.split("\t");
				phrss.add(ps[0]);
			}

			pm = new PhraseMapper<Integer>(PhraseMapper.createDict(phrss, ds.getVocab()));
		}

		List<String> Qs = Generics.newArrayList(bqs.size());

		for (BaseQuery bq : bqs) {
			Qs.add(bq.getSearchText());
			qData.add(ds.index(bq.getSearchText()));

			SparseVector Q = ds.index(bq.getSearchText());

			List<Integer> input = Generics.newArrayList(Q.size());

			for (int w : Q.indexes()) {
				input.add(w);
			}

			List<Pair<Integer, Integer>> ps = pm.map(input);

			if (ps.size() > 0) {
				System.out.println(bq.toString());

				for (Pair<Integer, Integer> p : ps) {
					int s = p.getFirst();
					int e = p.getSecond();
					StringBuffer sb = new StringBuffer();
					sb.append(String.format("[%d-%d,", s, e));

					for (int j = s; j < e; j++) {
						int w = input.get(j);
						String word = ds.getVocab().getObject(w);
						sb.append(" ");
						sb.append(word);
					}
					sb.append("]");
					System.out.println(sb.toString());
				}
				System.out.println();
			}

		}

		{
			List<SparseVector> res = ds.search(bqs, 2);
			dData.addAll(res);
		}

		collectSearchResults(ds, bqs, dData, srData, top_k);

		FileUtils.deleteFilesUnder(resDir);

		Performance p = PerformanceEvaluator.evalute(srData, relvData);

		p.writeObject(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		FileUtils.writeStringCounterMap(resDir + String.format("%s_%s.ser.gz", "sr", modelName), srData);
		new SparseMatrix(qData).writeObject(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		new SparseMatrix(dData).writeObject(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		System.out.println(p);
	}

	public void runKLDFBLemma() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		Performance p1 = new Performance(dataDir + "res/p_kld.ser.gz");

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		SparseMatrix qData1 = new SparseMatrix(dataDir + "res/q_kld-lemma.ser.gz");
		SparseMatrix dData1 = new SparseMatrix(dataDir + "res/d_kld-lemma.ser.gz");

		CounterMap<String, String> srData = Generics.newCounterMap();

		List<SparseVector> qData2 = Generics.newArrayList();
		List<SparseVector> dData2 = Generics.newArrayList();

		for (int j = 0; j < bqs.size(); j++) {
			BaseQuery bq = bqs.get(j);

			System.out.println(bq);

			SparseVector lm_q1 = qData1.rowAt(j);
			SparseVector scores1 = dData1.rowAt(j);
			scores1 = scores1.subVector(top_k);

			SparseVector lm_fb = ds.getFeedbackBuilder().buildRM1(scores1, 0);
			SparseVector lm_q2 = ds.updateQueryModel(lm_q1, lm_fb);

			SparseVector scores2 = ds.search(lm_q2);
			scores2 = scores2.subVector(top_k);

			qData2.add(lm_q2);
			dData2.add(scores2);

			collectSearchResults(ds, bq, scores2, srData, top_k);
		}

		String pFileName = dataDir + "res/p_kld-fb-lemma.ser.gz";
		String srFileName = dataDir + "res/sr_kld-fb-lemma.ser.gz";
		String qFileName = dataDir + "res/q_kld-fb-lemma.ser.gz";
		String dFileName = dataDir + "res/d_kld-fb-lemma.ser.gz";

		Performance p2 = PerformanceEvaluator.evalute(srData, relvData);
		p2.writeObject(pFileName);

		PerformanceComparator.compare(p1, p2);

		FileUtils.writeStringCounterMap(srFileName, srData);

		new SparseMatrix(qData2).writeObject(qFileName);
		new SparseMatrix(dData2).writeObject(dFileName);

		System.out.println(p2);
	}

	public void runKLDFBTmp() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		Vocab vocab = ds.getVocab();

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		SparseMatrix qData1 = new SparseMatrix(dataDir + "res/q_kld.ser.gz");
		SparseMatrix dData1 = new SparseMatrix(dataDir + "res/d_kld.ser.gz");

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser");
		WordSearcher ws = new WordSearcher(ds.getVocab(), E, null);

		CounterMap<String, String> srData = Generics.newCounterMap();

		String pFileName = dataDir + "res/p_kld-fb-tmp.ser.gz";
		String srFileName = dataDir + "res/sr_kld-fb-tmp.ser.gz";
		String qFileName = dataDir + "res/q_kld-fb-tmp.ser.gz";
		String dFileName = dataDir + "res/d_kld-fb-tmp.ser.gz";

		List<SparseVector> qData2 = Generics.newArrayList();
		List<SparseVector> dData2 = Generics.newArrayList();

		TranslationModelScorer depScorer = new TranslationModelScorer(vocab, ds.getDocumentCollection(), ds.getInvertedIndex(), ws,
				ds.getWordFilter());

		// ds.getFeedbackBuilder().setUseDocumentPrior(true);

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(bq);

			SparseVector lm_q1 = qData1.rowAt(i);
			SparseVector scores1 = dData1.rowAt(i);

			// lm_q1 = ds.getQueryModel(lm_q1, true);
			// dScores1 = ds.search(lm_q1);

			scores1 = scores1.subVector(top_k);

			// scores1 = depScorer.score(lm_q1, scores1);

			SparseVector lm_q2 = lm_q1;
			SparseVector scores2 = scores1;

			System.out.printf("lm_q\t%s\n", VectorUtils.toCounter(lm_q1, vocab));

			double mixture_fb = 0.5;

			for (int j = 0; j < 1; j++) {
				ds.setFeedbackMixture(mixture_fb);

				// SparseVector lm_fb =
				// ds.getFeedbackBuilder().buildERM1(scores2);
				// SparseVector lm_fb =
				// ds.getFeedbackBuilder().buildEmbeddingRM1(lm_q2, scores2,
				// ws);
				SparseVector lm_fb = ds.getFeedbackBuilder().buildEmbeddingRM2(lm_q2, scores2, ws);

				lm_q2 = ds.updateQueryModel(lm_q2, lm_fb);
				scores2 = ds.search(lm_q2);

				System.out.printf("lm_q\t%s\n", VectorUtils.toCounter(lm_q2, vocab));

				scores2 = scores2.subVector(top_k);

			}
			System.out.println();

			// SparseVector lm_fb = ds.getFeedbackBuilder().buildRM1(scores1,
			// 0);
			// SparseVector lm_q2 = ds.updateQueryModel(lm_q1, lm_fb);
			//
			// SparseVector scores2 = ds.search(lm_q2);
			// scores2 = scores2.subVector(top_k);
			//
			// SparseVector lm_fb2 = ds.getFeedbackBuilder().buildRM1(scores2,
			// 0);
			// SparseVector lm_q3 = ds.updateQueryModel(lm_q2, lm_fb2);
			//
			// SparseVector scores3 = ds.search(lm_q3);
			// scores3 = scores3.subVector(top_k);
			//
			// System.out.printf("lm_q1\t%s", VectorUtils.toCounter(lm_q1,
			// vocab));
			// System.out.printf("\nlm_q2\t%s", VectorUtils.toCounter(lm_q2,
			// vocab));
			// System.out.printf("\nlm_q3\t%s\n\n", VectorUtils.toCounter(lm_q3,
			// vocab));

			qData2.add(lm_q2);
			dData2.add(scores2);

			collectSearchResults(ds, bq, scores2, srData, top_k);
		}

		Performance p2 = PerformanceEvaluator.evalute(srData, relvData);
		p2.writeObject(pFileName);

		FileUtils.writeStringCounterMap(srFileName, srData);

		new SparseMatrix(qData2).writeObject(qFileName);
		new SparseMatrix(dData2).writeObject(dFileName);

		System.out.println(p2);

	}

	public void runKLDFBTune() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		Performance p1 = new Performance(dataDir + "res/p_kld.ser.gz");

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		SparseMatrix qData1 = new SparseMatrix(dataDir + "res/q_kld.ser.gz");
		SparseMatrix dData1 = new SparseMatrix(dataDir + "res/d_kld.ser.gz");

		FeedbackBuilder fb = ds.getFeedbackBuilder();

		int[] sizes_word_fb = { 30 };
		int[] sizes_doc_fb = { 5, 10, 20, 30, 40, 50 };

		for (int size_doc_fb : sizes_doc_fb) {
			for (int size_word_fb : sizes_word_fb) {
				fb.setFbDocSize(size_doc_fb);
				fb.setFbWordSize(size_word_fb);

				List<SparseVector> qData2 = Generics.newArrayList();
				List<SparseVector> dData2 = Generics.newArrayList();

				CounterMap<String, String> resData = Generics.newCounterMap();

				for (int j = 0; j < bqs.size(); j++) {
					BaseQuery bq = bqs.get(j);

					SparseVector lm_q1 = qData1.rowAt(j);
					SparseVector scores1 = dData1.rowAt(j);
					scores1 = scores1.subVector(top_k);

					SparseVector lm_fb = ds.getFeedbackBuilder().buildRM1(scores1, 0);
					SparseVector lm_q2 = ds.updateQueryModel(lm_q1, lm_fb);

					SparseVector scores2 = ds.search(lm_q2);
					scores2 = scores2.subVector(top_k);

					qData2.add(lm_q2);
					dData2.add(scores2);

					collectSearchResults(ds, bq, scores2, resData, top_k);
				}

				String pFileName = dataDir + "res/p_kld-fb.ser.gz";
				String srFileName = dataDir + "res/sr_p_kld-fb.ser.gz";
				String qFileName = dataDir + "res/q_kld-fb.ser.gz";
				String dFileName = dataDir + "res/d_p_kld-fb.ser.gz";

				Performance p2 = PerformanceEvaluator.evalute(resData, relvData);
				// p2.writeObject(pFileName);

				PerformanceComparator.compare(p1, p2);

				// FileUtils.writeStringCounterMap(srFileName, resDir);
				//
				// new SparseMatrix(qData2).writeObject(qFileName);
				// new SparseMatrix(dData2).writeObject(dFileName);

				System.out.printf("size_doc_fb:\t%d\n", size_doc_fb);
				System.out.printf("size_word_fb:\t%d\n", size_word_fb);
				System.out.println(p2);
				System.out.println();
			}
		}
	}

	public void runKLDLemma() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		Vocab vocab = ds.getVocab();

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> srData = Generics.newCounterMap();

		List<SparseVector> qData = Generics.newArrayList();
		List<SparseVector> dData = Generics.newArrayList();

		LemmaExpander le = new LemmaExpander(vocab, FileUtils.readStringHashMapFromText(MIRPath.TREC_CDS_2014_DIR + "lemmas.txt"));

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(bq);

			SparseVector Q = ds.index(bq.getSearchText());

			SparseVector lm_q0 = Q.copy();
			lm_q0.normalize();

			SparseVector lm_q1 = le.expand(lm_q0);
			lm_q1.normalizeAfterSummation();

			SparseVector scores = ds.search(lm_q1);

			scores = scores.subVector(top_k);

			qData.add(lm_q1);
			dData.add(scores);

			collectSearchResults(ds, bq, scores, srData, top_k);
		}

		Performance p = PerformanceEvaluator.evalute(srData, relvData);
		p.writeObject(dataDir + "res/p_kld-lemma.ser.gz");

		FileUtils.writeStringCounterMap(dataDir + "res/sr_kld-lemma.ser.gz", srData);

		new SparseMatrix(qData).writeObject(dataDir + "res/q_kld-lemma.ser.gz");
		new SparseMatrix(dData).writeObject(dataDir + "res/d_kld-lemma.ser.gz");

		System.out.println(p);
	}

	public void runKLDMetaMap() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		Vocab vocab = ds.getVocab();

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		Map<String, String> semMap = MetaMapUtils.readSemanticTypes(MIRPath.DATA_DIR + "metamap/SemanticTypes_2013AA.txt");
		Map<String, String> groupMap = MetaMapUtils.readSemanticGroups(MIRPath.DATA_DIR + "metamap/SemGroups_2013.txt");

		Set<String> toKeepSemTypes = Generics.newHashSet();
		toKeepSemTypes.add("Clinical Drug");
		toKeepSemTypes.add("Disease or Syndrome");
		toKeepSemTypes.add("Injury or Poisoning");
		toKeepSemTypes.add("Sign or Symptom");
		toKeepSemTypes.add("Therapeutic or Preventive Procedure");
		toKeepSemTypes.add("Finding");
		toKeepSemTypes.add("Anatomical Abnormality");

		List<String> mmqs = Generics.newArrayList();

		for (String s : FileUtils.readFromText(dataDir + "query_metamap.txt").split("\n\n")) {
			String[] lines = s.split("\n");

			List<String> items = Generics.newArrayList();

			for (int i = 6; i < lines.length - 1; i++) {
				String line = lines[i];
				String[] parts = StrUtils.unwrap(line.split("\t"));
				String phr = parts[0];

				String cpt = parts[4];
				String[] semTypeShorts = parts[5].split("\\|");
				String pref = parts[6];
				String mWords = parts[7];

				boolean flag = false;

				for (String semTypeShort : semTypeShorts) {
					String semTypeLong = semMap.get(semTypeShort);
					String groupPath = groupMap.get(semTypeLong);

					if (toKeepSemTypes.contains(semTypeLong)) {
						flag = true;
						break;
					}
				}

				if (flag) {
					items.add(cpt);
					items.add(pref);
				}

			}
			mmqs.add(StrUtils.join("  ", items));
		}

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> srData = Generics.newCounterMap();

		List<SparseVector> qData = Generics.newArrayList();
		List<SparseVector> dData = Generics.newArrayList();

		SimpleStringNormalizer sn = new SimpleStringNormalizer(true);

		LemmaExpander le = new LemmaExpander(vocab, FileUtils.readStringHashMapFromText(MIRPath.TREC_CDS_2014_DIR + "lemmas.txt"));

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			String mq = mmqs.get(i);

			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(bq);

			String st = StrUtils.join(" ", NLPUtils.tokenize(tcq.getSearchText(false)));
			st = sn.normalize(st);

			SparseVector Q = ds.index(st);

			SparseVector lm_q0 = Q.copy();
			SparseVector lm_q1 = lm_q0.copy();
			SparseVector lm_mm = new SparseVector();

			{
				String st2 = StrUtils.join(" ", NLPUtils.tokenize(mq));
				st2 = sn.normalize(st2).trim();

				Counter<String> c = Generics.newCounter();

				if (st2.length() > 0) {
					for (String word : st2.split(" ")) {
						if (!ds.getWordFilter().filter(word)) {
							c.setCount(word, 1);
							// c.incrementCount(word, 1);
						}
					}
				}

				lm_mm = VectorUtils.toSparseVector(c, vocab);
				lm_mm.normalize();

				System.out.println(c.toString());
			}

			SparseVector lm_q2 = le.expand(lm_q1);
			lm_q2.normalizeAfterSummation();

			double mixture = 0.1;

			lm_q2 = VectorMath.addAfterMultiply(lm_q2, 1 - mixture, lm_mm, mixture);
			lm_q2.normalize();

			lm_q0.normalize();
			lm_q1.normalize();

			System.out.printf("lm_q: %s\n", VectorUtils.toCounter(lm_q1, vocab));
			System.out.printf("lm_q: %s\n", VectorUtils.toCounter(lm_q2, vocab));
			System.out.printf("lm_mm: %s\n", VectorUtils.toCounter(lm_mm, vocab));

			SparseVector scores = ds.search(lm_q2);

			scores = scores.subVector(top_k);

			qData.add(lm_q2);
			dData.add(scores);

			collectSearchResults(ds, bq, scores, srData, top_k);
		}

		Performance p = PerformanceEvaluator.evalute(srData, relvData);
		p.writeObject(dataDir + "res/p_kld-mm.ser.gz");

		FileUtils.writeStringCounterMap(dataDir + "res/sr_kld-mm.ser.gz", srData);

		new SparseMatrix(qData).writeObject(dataDir + "res/q_kld-mm.ser.gz");
		new SparseMatrix(dData).writeObject(dataDir + "res/d_kld-mm.ser.gz");

		System.out.println(p);
	}

	public void runKLDTmp() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		Vocab vocab = ds.getVocab();

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> resData = Generics.newCounterMap();

		SparseMatrix qData1 = new SparseMatrix(dataDir + "res/q_kld-lemma.ser.gz");
		SparseMatrix dData1 = new SparseMatrix(dataDir + "res/d_kld-lemma.ser.gz");

		List<SparseVector> qData2 = Generics.newArrayList();
		List<SparseVector> dData2 = Generics.newArrayList();

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser", true);
		WordSearcher ws = new WordSearcher(ds.getVocab(), E, null);

		LemmaExpander le = new LemmaExpander(vocab, FileUtils.readStringHashMapFromText(MIRPath.TREC_CDS_2014_DIR + "lemmas.txt"));

		DenseVector cosineSums = new DenseVector(vocab.size());

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(bq);

			SparseVector lm_q1 = qData1.rowAt(i);
			SparseVector scores1 = dData1.rowAt(i);

			// System.out.println(VectorUtils.toCounter(c, vocab));

			// for (int j = 0; j < Q.size(); j++) {
			// int w = Q.indexAt(j);
			// String word = vocab.getObject(w);
			//
			// SparseVector rels1 = ws.getRelatedWords(vocab.indexOf("man"),
			// vocab.indexOf("boy"), w, 20);
			// SparseVector rels2 = ws.getRelatedWords(vocab.indexOf("woman"),
			// vocab.indexOf("girl"), w, 20);
			// SparseVector rels3 = VectorMath.addAfterMultiply(rels1, 0.5,
			// rels2, 0.5);
			//
			// for (int k = 0; k < rels3.size(); k++) {
			// int w2 = rels3.indexAt(k);
			// double rel = rels3.valueAt(k);
			//
			// if (ds.getWordFilter().filter(w2) || rel < 0.8) {
			// continue;
			// }
			//
			// c.incrementCount(w2, 1);
			// }
			// // System.out.printf("rels1: %s, %s\n", word,
			// VectorUtils.toCounter(rels1, vocab));
			// // System.out.printf("rels2: %s, %s\n", word,
			// VectorUtils.toCounter(rels2, vocab));
			// // System.out.printf("rels3: %s, %s\n", word,
			// VectorUtils.toCounter(rels3, vocab));
			// }

			// Counter<Integer> c0 = VectorUtils.toCounter(Q);
			//
			// for (int w : c.keySet()) {
			// c0.incrementCount(w, 0.5);
			// }
			//

			// for (int j = 0; j < lm_q1.size(); j++) {
			// int w1 = lm_q1.indexAt(j);
			//
			// if (cosineSums.value(w1) == 0) {
			// for (int w2 = 0; w2 < vocab.size(); w2++) {
			// double cosine = ws.getCosine(w1, w2);
			// cosine = CommonMath.sigmoid(cosine);
			// cosineSums.add(w1, cosine);
			// }
			// }
			// }

			DenseMatrix cosines = new DenseMatrix(lm_q1.size());

			for (int j = 0; j < lm_q1.size(); j++) {
				int w1 = lm_q1.indexAt(j);
				double idf1 = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w1));
				double pr1 = vocab.getProb(w1);
				double cosine_sum = cosineSums.value(w1);

				for (int k = j + 1; k < lm_q1.size(); k++) {
					int w2 = lm_q1.indexAt(k);
					double idf2 = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w2));
					double pr2 = vocab.getProb(w2);
					double cosine = ws.getCosine(w1, w2);

					// cosine = Math.exp(10 * cosine);

					if (cosine < 0) {
						cosine = 0;
					}

					cosine = CommonMath.sigmoid(cosine);
					// cosine /= cosine_sum;

					// cosine *= pr1 * pr2;

					// cosine_sum = cosineSums.value(index)
					cosine = cosine * idf1 * idf2;
					cosines.add(j, k, cosine);
					cosines.add(k, j, cosine);
				}
			}

			cosines.normalizeColumns();

			SparseVector lm_q2 = lm_q1.copy();
			lm_q2.normalize();

			for (int j = 0; j < lm_q1.size(); j++) {
				int w1 = lm_q1.indexAt(j);
				double pr_w1_in_q = lm_q1.valueAt(j);
				// double pr_w1_in_q_emb = VectorMath.dotProduct(lm_q0,
				// cosines.row(j));
				double pr_w1_in_q_emb = ArrayMath.dotProduct(lm_q1.values(), cosines.row(j).values());

				String word = vocab.getObject(w1);

				lm_q2.setAt(j, pr_w1_in_q_emb);

				// System.out.printf("%d, %s, %f, %f, %f\n", j, word,
				// pr_w1_in_q, pr_w1_in_q_emb, pr_w1_in_q_emb / pr_w1_in_q);
			}

			lm_q2.normalizeAfterSummation();

			// System.out.println();

			ArrayMath.randomWalk(cosines.values(), lm_q2.values(), 30, 0.0000001, 1);

			lm_q2.normalizeAfterSummation();

			// SparseVector lm_q1 = new SparseVector(c0);
			// lm_q1.normalize();
			//
			// SparseVector lm_q2 = le.expand(lm_q1);
			// lm_q2.normalize();
			//
			// DenseMatrix m = new DenseMatrix(lm_q2.size());
			//
			// for (int j = 0; j < lm_q2.size(); j++) {
			// int w1 = lm_q2.indexAt(j);
			// double pr1 = vocab.getProb(w1);
			// double doc_freq1 = vocab.getDocFreq(w1);
			// double idf1 = TermWeighting.idf(vocab.getDocCnt(),
			// vocab.getDocFreq(w1));
			//
			// for (int k = j + 1; k < lm_q2.size(); k++) {
			// int w2 = lm_q2.indexAt(k);
			// double pr2 = vocab.getProb(w2);
			// double idf2 = TermWeighting.idf(vocab.getDocCnt(),
			// vocab.getDocFreq(w2));
			// double cosine = ws.getCosine(w1, w2);
			//
			// if (cosine > 0) {
			// m.add(j, k, cosine * idf1 * idf2);
			// m.add(k, j, cosine * idf1 * idf2);
			// }
			// }
			// }
			//
			// m.normalizeColumns();
			//
			// SparseVector lm_q3 = lm_q2.copy();
			//
			// ArrayMath.randomWalk(m.values(), lm_q3.values(), 20);
			//
			// SparseVector lm_fb = new SparseVector(c);
			//

			SparseVector lm_q3 = ds.updateQueryModel(lm_q1, lm_q2);

			System.out.printf("lm_q1\t%s", VectorUtils.toCounter(lm_q1, vocab));
			System.out.printf("\nlm_q2\t%s", VectorUtils.toCounter(lm_q2, vocab));
			System.out.printf("\nlm_q3\t%s", VectorUtils.toCounter(lm_q3, vocab));
			// System.out.printf("\nlm_fb\t%s\n\n", VectorUtils.toCounter(lm_fb,
			// vocab));
			System.out.println();
			System.out.println();

			SparseVector scores2 = ds.search(lm_q3);
			// scores = ds.search(lm_q, scores);
			scores2 = scores2.subVector(top_k);

			qData2.add(lm_q3);
			dData2.add(scores2);

			collectSearchResults(ds, bq, scores2, resData, top_k);
		}

		Performance p = PerformanceEvaluator.evalute(resData, relvData);
		p.writeObject(dataDir + "res/p_kld-tmp.ser.gz");

		FileUtils.writeStringCounterMap(dataDir + "res/sr_kld-tmp.ser.gz", resData);

		new SparseMatrix(qData2).writeObject(dataDir + "res/q_kld-lemma-2.ser.gz");
		new SparseMatrix(dData2).writeObject(dataDir + "res/d_kld-lemma-2.ser.gz");

		System.out.println(p);
	}

	public void runKLDTune() throws Exception {
		DocumentSearcher ds1 = new DocumentSearcher(idxDir, stopwordFileName);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> resData1 = Generics.newCounterMap();

		LMScorer scorer = (LMScorer) ds1.getScorer();
		// scorer.setQueryBackgroundModel(lm_qbg);

		// double[] mixture_jms = { 0.1, 0.3, 0.5, 0.7, 0.9 };
		double[] mixtures_jm = { 0, 0.5 };
		double[] priors_dir = { 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };

		SparseMatrix qData1 = new SparseMatrix();
		SparseMatrix dData1 = new SparseMatrix();

		qData1.readObject(dataDir + "res/q_kld.ser.gz");
		dData1.readObject(dataDir + "res/d_kld.ser.gz");

		for (int i = 0; i < mixtures_jm.length; i++) {
			for (int j = 0; j < priors_dir.length; j++) {
				scorer.setJMMixture(mixtures_jm[i]);
				scorer.setDirichletPrior(priors_dir[j]);
				for (int k = 0; k < bqs.size(); k++) {
					BaseQuery bq = bqs.get(k);
					SparseVector Q1 = qData1.rowAt(k);
					SparseVector lm_q1 = ds1.getQueryModel(Q1);
					SparseVector scores = ds1.search(lm_q1);
					scores = scores.subVector(top_k);

					collectSearchResults(ds1, bq, scores, resData1, top_k);
				}

				Performance p1 = PerformanceEvaluator.evalute(resData1, relvData);

				System.out.printf("mixture_jm:\t%f\n", mixtures_jm[i]);
				System.out.printf("prior_dirichlet:\t%f\n", priors_dir[j]);
				System.out.println(p1);
				System.out.println();
			}
		}
	}

	public void runThreeStageEEM() throws Exception {
		DocumentSearcher ds1 = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		DocumentSearcher ds2 = new DocumentSearcher(MIRPath.WIKI_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		DocumentSearcher ds3 = new DocumentSearcher(MIRPath.MESH_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(MIRPath.TREC_CDS_2014_QUERY_FILE);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(MIRPath.TREC_CDS_2014_REL_JUDGE_FILE);

		int top_k = 1000;

		boolean[] flags1 = { false, true };
		boolean[] flags2 = { false, true };
		boolean[] flags3 = { false, true };
		boolean[] flags4 = { false, true };

		for (Boolean f1 : flags1) {
			for (Boolean f2 : flags2) {
				for (Boolean f3 : flags3) {
					for (Boolean f4 : flags4) {
						CounterMap<String, String> resData = Generics.newCounterMap();
						List<SparseVector> qs = Generics.newArrayList();

						for (int i = 0; i < bqs.size(); i++) {
							BaseQuery bq = bqs.get(i);
							TrecCdsQuery tcq = (TrecCdsQuery) bq;

							System.out.println(bq);

							SparseVector Q1 = ds1.index(bq.getSearchText());
							SparseVector Q2 = ds2.index(bq.getSearchText());
							SparseVector Q3 = ds3.index(bq.getSearchText());

							SparseVector lm_q1 = ds1.getQueryModel(Q1);
							SparseVector lm_q2 = ds2.getQueryModel(Q2);
							SparseVector lm_q3 = ds3.getQueryModel(Q3);

							SparseVector docScores1 = ds1.search(lm_q1);
							SparseVector docScores2 = ds2.search(lm_q2);
							SparseVector docScores3 = ds3.search(lm_q3);

							List<DocumentSearcher> searchers = Generics.newArrayList();
							searchers.add(ds1);
							searchers.add(ds2);
							searchers.add(ds3);

							List<SparseVector> docScoreData = Generics.newArrayList();
							docScoreData.add(docScores1);
							docScoreData.add(docScores2);
							docScoreData.add(docScores3);

							SparseVector lm_fb = ds1.getFeedbackBuilder().buildEEM(searchers, docScoreData);
							SparseVector lm_q4 = ds1.updateQueryModel(lm_q1, lm_fb);

							docScores1 = ds1.search(lm_q4);

							collectSearchResults(ds1, bq, docScores1, resData, top_k);

							qs.add(lm_q4);
						}

						Performance p = PerformanceEvaluator.evalute(resData, relvData);

						StringBuffer sb = new StringBuffer();
						for (Boolean f : new Boolean[] { f1, f2, f3, f4 }) {
							sb.append(f.toString().charAt(0) + "");
						}

						String outFileName1 = String.format(MIRPath.TREC_CDS_2014_DIR + "/res/perf_kld-eem-mesh-%s.ser.gz", sb.toString());
						String outFileName2 = String.format(MIRPath.TREC_CDS_2014_DIR + "/res/sr_kld-eem-mesh-%s.ser.gz", sb.toString());
						String outFileName3 = String.format(MIRPath.TREC_CDS_2014_DIR + "/res/q_kld-eem-mesh-%s.ser.gz", sb.toString());

						p.writeObject(outFileName1);
						FileUtils.writeStringCounterMap(outFileName2, resData);

						SparseMatrix Q = new SparseMatrix(qs);
						Q.writeObject(outFileName3);

						System.out.println(outFileName1);
						System.out.println(p);
						System.out.println();
					}
				}
			}
		}
	}

	public void runThreeStageEEMWithMeSH() throws Exception {
		DocumentSearcher ds1 = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		DocumentSearcher ds2 = new DocumentSearcher(MIRPath.WIKI_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		DocumentSearcher ds3 = new DocumentSearcher(MIRPath.MESH_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(MIRPath.TREC_CDS_2014_QUERY_FILE);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(MIRPath.TREC_CDS_2014_REL_JUDGE_FILE);

		int top_k = 1000;

		boolean[] flags1 = { false, true };
		boolean[] flags2 = { false, true };
		boolean[] flags3 = { false, true };
		boolean[] flags4 = { false, true };

		for (Boolean f1 : flags1) {
			for (Boolean f2 : flags2) {
				for (Boolean f3 : flags3) {
					for (Boolean f4 : flags4) {
						CounterMap<String, String> resData = Generics.newCounterMap();
						List<SparseVector> qs = Generics.newArrayList();

						for (int i = 0; i < bqs.size(); i++) {
							BaseQuery bq = bqs.get(i);
							TrecCdsQuery tcq = (TrecCdsQuery) bq;

							System.out.println(bq);

							SparseVector Q1 = ds1.index(bq.getSearchText());
							SparseVector Q2 = ds2.index(bq.getSearchText());

							SparseVector lm_q1 = ds1.getQueryModel(Q1);
							SparseVector lm_q2 = ds2.getQueryModel(Q2);

							SparseVector docScores1 = ds1.search(lm_q1);
							SparseVector docScores2 = ds2.search(lm_q2);

							List<DocumentSearcher> searchers = Generics.newArrayList();
							searchers.add(ds1);
							searchers.add(ds2);

							List<SparseVector> docScoreData = Generics.newArrayList();
							docScoreData.add(docScores1);
							docScoreData.add(docScores2);

							SparseVector lm_fb = ds1.getFeedbackBuilder().buildEEM(searchers, docScoreData);
							lm_q1 = ds1.updateQueryModel(lm_q1, lm_fb);

							SparseVector lm_q3 = VectorUtils.toSparseVector(VectorUtils.toCounter(lm_q1, ds1.getVocab()), ds3.getVocab());
							lm_q3.normalize();

							docScores1 = ds1.search(lm_q1);
							docScores2 = ds3.search(lm_q3);

							searchers = Generics.newArrayList();
							searchers.add(ds1);
							searchers.add(ds3);

							docScoreData = Generics.newArrayList();
							docScoreData.add(docScores1);
							docScoreData.add(docScores2);

							collectSearchResults(ds1, bq, docScores1, resData, top_k);

							qs.add(lm_q1);
						}

						Performance p = PerformanceEvaluator.evalute(resData, relvData);

						StringBuffer sb = new StringBuffer();
						for (Boolean f : new Boolean[] { f1, f2, f3, f4 }) {
							sb.append(f.toString().charAt(0) + "");
						}

						String outFileName1 = String.format(MIRPath.TREC_CDS_2014_DIR + "/res/perf_kld-eem-mesh-%s.ser.gz", sb.toString());
						String outFileName2 = String.format(MIRPath.TREC_CDS_2014_DIR + "/res/sr_kld-eem-mesh-%s.ser.gz", sb.toString());
						String outFileName3 = String.format(MIRPath.TREC_CDS_2014_DIR + "/res/q_kld-eem-mesh-%s.ser.gz", sb.toString());

						p.writeObject(outFileName1);
						FileUtils.writeStringCounterMap(outFileName2, resData);

						SparseMatrix Q = new SparseMatrix(qs);
						Q.writeObject(outFileName3);

						System.out.println(outFileName1);
						System.out.println(p);
						System.out.println();
					}
				}
			}
		}
	}

	public void runTwoStageCBEEM() throws Exception {
		DocumentSearcher ds1 = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		DocumentSearcher ds2 = new DocumentSearcher(MIRPath.WIKI_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		Vocab vocab = ds1.getVocab();

		ds1.getFeedbackBuilder().setMixtureJM(0.8);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(MIRPath.TREC_CDS_2014_QUERY_FILE);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(MIRPath.TREC_CDS_2014_REL_JUDGE_FILE);

		int top_k = 1000;

		boolean[] flags1 = { false };
		boolean[] flags2 = { false };
		boolean[] flags3 = { false };

		for (int l1 = 0; l1 < flags1.length; l1++) {
			for (int l2 = 0; l2 < flags2.length; l2++) {
				for (int l3 = 0; l3 < flags3.length; l3++) {
					Boolean f1 = flags1[l1];
					Boolean f2 = flags2[l2];
					Boolean f3 = flags3[l3];

					CounterMap<String, String> resData = Generics.newCounterMap();
					List<SparseVector> qs = Generics.newArrayList();

					for (int i = 0; i < bqs.size(); i++) {
						BaseQuery bq = bqs.get(i);
						TrecCdsQuery tcq = (TrecCdsQuery) bq;

						System.out.println(bq);

						SparseVector Q1 = ds1.index(bq.getSearchText());
						SparseVector Q2 = ds2.index(bq.getSearchText());

						SparseVector lm_q1 = ds1.getQueryModel(Q1);
						SparseVector lm_q2 = ds2.getQueryModel(Q2);

						SparseVector docScores1 = ds1.search(lm_q1);
						SparseVector docScores2 = ds2.search(lm_q2);

						List<DocumentSearcher> dss1 = Generics.newArrayList();
						dss1.add(ds1);
						dss1.add(ds2);

						List<SparseVector> dss2 = Generics.newArrayList();
						dss2.add(docScores1);
						dss2.add(docScores2);

						SparseVector lm_fb = ds1.getFeedbackBuilder().buildCBEEM(dss1, dss2);

						SparseVector lm_q3 = ds1.updateQueryModel(lm_q1, lm_fb);

						docScores1 = ds1.search(lm_q3);

						collectSearchResults(ds1, bq, docScores1, resData, top_k);

						qs.add(lm_q3);

						System.out.printf("lm_q1\t%s", VectorUtils.toCounter(lm_q1, vocab));
						System.out.printf("\nlm_q2\t%s", VectorUtils.toCounter(lm_q2, vocab));
						System.out.printf("\nlm_q3\t%s\n\n", VectorUtils.toCounter(lm_q3, vocab));
					}

					Performance p = PerformanceEvaluator.evalute(resData, relvData);

					StringBuffer sb = new StringBuffer();
					for (Boolean f : new Boolean[] { f1, f2, f3 }) {
						sb.append(f.toString().charAt(0) + "");
					}

					String outFileName1 = String.format(MIRPath.TREC_CDS_2014_DIR + "/res/perf_kld-cbeem-%s.ser.gz", sb.toString());
					String outFileName2 = String.format(MIRPath.TREC_CDS_2014_DIR + "/res/sr_kld-cbeem-%s.ser.gz", sb.toString());
					String outFileName3 = String.format(MIRPath.TREC_CDS_2014_DIR + "/res/q_kld-cbeem-%s.ser.gz", sb.toString());

					p.writeObject(outFileName1);
					FileUtils.writeStringCounterMap(outFileName2, resData);

					SparseMatrix Q = new SparseMatrix(qs);
					Q.writeObject(outFileName3);

					System.out.println(outFileName1);
					System.out.println(p);
					System.out.println();
				}
			}
		}
	}

	public void runTwoStageEEM() throws Exception {
		DocumentSearcher ds1 = new DocumentSearcher(idxDir, stopwordFileName);
		DocumentSearcher ds2 = new DocumentSearcher(MIRPath.WIKI_COL_INDEX_DIR, stopwordFileName);

		Vocab vocab1 = ds1.getVocab();
		Vocab vocab2 = ds2.getVocab();

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		boolean[] flags1 = { true };
		boolean[] flags2 = { false };
		boolean[] flags3 = { false, true };

		SparseMatrix qData0 = new SparseMatrix(dataDir + "res/q_kld.ser.gz");
		SparseMatrix qData1 = new SparseMatrix(dataDir + "res/q_kld.ser.gz");
		SparseMatrix dData1 = new SparseMatrix(dataDir + "res/d_kld-fb-lemma.ser.gz");

		for (Boolean f1 : flags1) {
			for (Boolean f2 : flags2) {
				for (Boolean f3 : flags3) {
					CounterMap<String, String> srData = Generics.newCounterMap();
					List<SparseVector> qData2 = Generics.newArrayList();
					List<SparseVector> dData2 = Generics.newArrayList();

					for (int i = 0; i < bqs.size(); i++) {
						BaseQuery bq = bqs.get(i);
						System.out.println(bq);

						SparseVector lm_q1 = qData1.rowAt(i);
						SparseVector lm_q2 = VectorUtils.toSparseVector(qData0.rowAt(i), vocab1, vocab2);

						lm_q2 = ds2.getQueryModel(lm_q2);

						SparseVector scores1 = dData1.rowAt(i);
						SparseVector scores2 = ds2.search(lm_q2);

						scores1 = scores1.subVector(top_k);
						scores2 = scores2.subVector(top_k);

						List<DocumentSearcher> dss1 = Generics.newArrayList();
						dss1.add(ds1);
						dss1.add(ds2);

						List<SparseVector> dss2 = Generics.newArrayList();
						dss2.add(scores1);
						dss2.add(scores2);

						SparseVector lm_fb = ds1.getFeedbackBuilder().buildEEM(dss1, dss2);

						SparseVector lm_q3 = ds1.updateQueryModel(lm_q1, lm_fb);

						SparseVector scores3 = ds1.search(lm_q3);
						scores3 = scores3.subVector(top_k);

						collectSearchResults(ds1, bq, scores3, srData, top_k);

						qData2.add(lm_q3);
						dData2.add(scores3);
					}

					Performance p = PerformanceEvaluator.evalute(srData, relvData);

					StringBuffer sb = new StringBuffer();
					for (Boolean f : new Boolean[] { f1, f2, f3 }) {
						sb.append(f.toString().charAt(0) + "");
					}

					String outFileName1 = String.format(dataDir + "/res/p_kld-eem-%s.ser.gz", sb.toString());
					String outFileName2 = String.format(dataDir + "/res/sr_kld-eem-%s.ser.gz", sb.toString());
					String outFileName3 = String.format(dataDir + "/res/q_kld-eem-%s.ser.gz", sb.toString());
					String outFileName4 = String.format(dataDir + "/res/d_kld-eem-%s.ser.gz", sb.toString());

					p.writeObject(outFileName1);
					FileUtils.writeStringCounterMap(outFileName2, srData);

					new SparseMatrix(qData2).writeObject(outFileName3);
					new SparseMatrix(dData2).writeObject(outFileName4);

					System.out.println(outFileName1);
					System.out.println(p);
					System.out.println();
				}
			}
		}
	}

	public void runVSM() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> srData = Generics.newCounterMap();

		List<SparseVector> qData = Generics.newArrayList();
		List<SparseVector> dData = Generics.newArrayList();

		DenseVector docNorms = null;

		String docNormFileName = idxDir + "doc_norm.ser.gz";

		if (FileUtils.exists(docNormFileName)) {
			docNorms = new DenseVector(docNormFileName);
		} else {
			docNorms = VsmScorer.getDocNorms(ds.getVocab(), ds.getDocumentCollection());
			docNorms.writeObject(docNormFileName);
		}

		VsmScorer scorer = new VsmScorer(ds);
		ds.setScorer(scorer);

		LemmaExpander le = new LemmaExpander(ds.getVocab(), FileUtils.readStringHashMapFromText(MIRPath.TREC_CDS_2014_DIR + "lemmas.txt"));

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(bq);

			SparseVector Q = ds.index(bq.getSearchText());

			// Q = le.expand(Q);

			SparseVector scores = ds.search(Q);

			qData.add(Q);
			dData.add(scores);

			collectSearchResults(ds, bq, scores, srData, top_k);
		}

		Performance p = PerformanceEvaluator.evalute(srData, relvData);
		// p.writeObject(dataDir + "res/p_bm25.ser.gz");
		//
		// FileUtils.writeStringCounterMap(dataDir + "res/sr_bm25.ser.gz",
		// srData);
		//
		// new SparseMatrix(qData).writeObject(dataDir + "res/q_bm25.ser.gz");
		// new SparseMatrix(dData).writeObject(dataDir + "res/d_bm25.ser.gz");

		System.out.println(p);
	}

	public void runWiki() throws Exception {
		DocumentSearcher ds1 = new DocumentSearcher(idxDir, stopwordFileName);
		DocumentSearcher ds2 = new DocumentSearcher(MIRPath.WIKI_COL_INDEX_DIR, stopwordFileName);

		List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(queryFileName);

		CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

		CounterMap<String, String> resData = Generics.newCounterMap();
		List<SparseVector> qData = Generics.newArrayList();
		List<SparseVector> dData = Generics.newArrayList();

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			TrecCdsQuery tcq = (TrecCdsQuery) bq;

			System.out.println(bq);

			SparseVector Q1 = ds1.index(bq.getSearchText());
			SparseVector Q2 = ds2.index(" disease diseases test tests treatment treatments");

			SparseVector lm_q1 = ds1.getQueryModel(Q1);
			SparseVector lm_q2 = ds2.getQueryModel(Q2);

			SparseVector scores1 = ds1.search(lm_q1);
			SparseVector scores2 = ds2.search(lm_q2);

			scores1 = scores1.subVector(top_k);
			scores2 = scores2.subVector(top_k);

			// List<DocumentSearcher> dss1 = Generics.newArrayList();
			// dss1.add(ds1);
			// dss1.add(ds2);
			//
			// List<SparseVector> dss2 = Generics.newArrayList();
			// dss2.add(scores1);
			// dss2.add(scores2);
			//
			// SparseVector lm_fb = ds1.getFeedbackBuilder().buildEEM(dss1,
			// dss2);
			//
			// // lm_q2 = VectorUtils.toSparseVector(lm_q2, vocab2, vocab1);
			// // lm_q2 = ds1.updateQueryModel(lm_q1, lm_q2);
			//
			// SparseVector lm_q3 = ds1.updateQueryModel(lm_q1, lm_fb);
			//
			// SparseVector scores3 = ds1.search(lm_q3);
			// scores3 = scores3.subVector(top_k);
			//
			// collectSearchResults(ds1, bq, scores3, resDir, top_k);
			//
			// qData.add(lm_q3);
			// dData.add(scores3);
		}

		Performance p = PerformanceEvaluator.evalute(resData, relvData);

		// StringBuffer sb = new StringBuffer();
		// for (Boolean f : new Boolean[] { f1, f2, f3 }) {
		// sb.append(f.toString().charAt(0) + "");
		// }

		// String outFileName1 = String.format(dataDir +
		// "/res/p_kld-eem-%s.ser.gz", sb.toString());
		// String outFileName2 = String.format(dataDir +
		// "/res/sr_kld-eem-%s.ser.gz", sb.toString());
		// String outFileName3 = String.format(dataDir +
		// "/res/q_kld-eem-%s.ser.gz", sb.toString());
		// String outFileName4 = String.format(dataDir +
		// "/res/d_kld-eem-%s.ser.gz", sb.toString());
		//
		// p.writeObject(outFileName1);
		// FileUtils.writeStringCounterMap(outFileName2, resDir);
		//
		// new SparseMatrix(qData).writeObject(outFileName3);
		// new SparseMatrix(dData).writeObject(outFileName4);
		//
		// System.out.println(outFileName1);
		// System.out.println(p);
		// System.out.println();
	}

}
