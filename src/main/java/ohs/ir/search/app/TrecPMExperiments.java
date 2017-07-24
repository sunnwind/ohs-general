package ohs.ir.search.app;

import java.io.File;
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
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.DocumentIdMap;
import ohs.corpus.type.EnglishNormalizer;
import ohs.corpus.type.EnglishTokenizer;
import ohs.corpus.type.StringNormalizer;
import ohs.corpus.type.StringTokenizer;
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
import ohs.ir.search.model.BM25Scorer;
import ohs.ir.search.model.DocumentPriorEstimator;
import ohs.ir.search.model.DocumentPriorEstimator.Type;
import ohs.ir.search.model.FeedbackBuilder;
import ohs.ir.search.model.LMScorer;
import ohs.ir.search.model.MRFScorer;
import ohs.ir.search.model.Scorer;
import ohs.ir.search.model.VSMScorer;
import ohs.ir.search.model.WeightedMRFScorer;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.types.generic.Pair;
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class TrecPMExperiments {

	public static int COR_TYPE = 1;

	public static String dataDir = MIRPath.TREC_PM_2017_DIR;

	public static String resDir = dataDir + "res/";

	public static String idxDir = MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR;

	public static String queryFileName = MIRPath.TREC_PM_2017_QUERY_FILE;

	public static String relFileName = "";

	public static String stopwordFileName = MIRPath.STOPWORD_INQUERY_FILE;

	static {
		if (COR_TYPE == 0) {
			dataDir = MIRPath.TREC_CDS_2016_DIR;
			idxDir = MIRPath.TREC_CDS_2016_COL_DC_DIR;
			queryFileName = MIRPath.TREC_CDS_2016_QUERY_FILE;
			relFileName = MIRPath.TREC_CDS_2016_REL_JUDGE_FILE;
		} else if (COR_TYPE == 1) {
			dataDir = MIRPath.TREC_PM_2017_DIR;
			idxDir = MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR;
			queryFileName = MIRPath.TREC_PM_2017_QUERY_FILE;
			relFileName = "";
		} else if (COR_TYPE == 2) {
			dataDir = MIRPath.TREC_PM_2017_DIR;
			idxDir = MIRPath.TREC_PM_2017_COL_CLINICAL_DC_DIR;
			queryFileName = MIRPath.TREC_PM_2017_QUERY_FILE;
			relFileName = "";
		}
		resDir = dataDir + "res/";
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		TrecPMExperiments e = new TrecPMExperiments();

		// e.getLemmas();

		// e.getRelevanceJudgeDocIds();

		// e.runPhraseMapping();
		// e.getMedicalDocPriors();
		// e.getQualityDocPriors();
		// e.getQualityDocPriors2();
		// e.getLengthDocPriors();
		// e.getIDFDocPriors();
		// e.getStopwordPriors();
		// e.getRelevanceFeedbacks();
		// e.buildRelevanceFeedbackQueries();
		// e.buildMedicalFeedbackQueries();
		// e.getSimMatrix();

		e.runInitSearch();
		// e.runReranking();
		// e.runReranking2();
		// e.runReranking3();

		// e.analyze1();
		// e.analyze2();

		// e.formatTrecOutput();

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
			SparseVector lm_fb = fb.buildRM1(scores);
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

	public void buildRelevanceFeedbackQueries() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);
		List<SparseVector> qData = Generics.newArrayList(bqs.size());

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		ds.setTopK(top_k);

		LMScorer scorer = (LMScorer) ds.getScorer();

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector Q = ds.index(bq.getSearchText());
			qData.add(Q);
		}

		DenseVector qualityPriors = new DenseVector(dataDir + "doc_prior_quality-2.ser.gz");

		FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(), ds.getWordFilter());
		fb.setFbWordSize(Integer.MAX_VALUE);

		Map<String, Integer> docIdMap = getDocumentIdMap();

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector Q = qData.get(i);
			Counter<String> c = relvData.getCounter(bq.getId());

			Counter<Integer> c1 = Generics.newCounter();
			Counter<Integer> c2 = Generics.newCounter();
			for (Entry<String, Double> e : c.entrySet()) {
				String did = e.getKey();
				double relv = e.getValue();
				int dseq = docIdMap.get(did);
				if (dseq < 0) {
					continue;
				}

				if (relv > 0) {
					c1.setCount(dseq, relv);
				} else {
					c2.setCount(dseq, 0);
				}
			}

			SparseVector rDocs = new SparseVector(c1);
			SparseVector nrDocs = new SparseVector(c2);

			SparseVector lm_q = Q.copy();
			lm_q.normalize();

			SparseVector rScores = scorer.scoreFromCollection(Q, rDocs);
			scorer.postprocess(rScores);

			for (int j = 0; j < rScores.size(); j++) {
				int dseq = rScores.indexAt(j);
				double score = rScores.valueAt(j);
				double relv = rDocs.value(dseq);
				// System.out.printf("[%s, %d, %d, %f, %f]\n", bq.getId(), j + 1, dseq, score,
				// relv);
				double score2 = score + relv;
				rScores.setAt(j, score2);
			}
			rScores.summation();
			rScores.sortValues();

			fb.setFbDocSize(rScores.size());
			SparseVector lm_fb = fb.buildRM1(rScores, 0, qualityPriors);
			SparseVector lm_q2 = fb.updateQueryModel(lm_q, lm_fb);
			qData.set(i, lm_q2);

			System.out.println(bq.toString());
			System.out.println(VectorUtils.toCounter(lm_fb, ds.getVocab()));
			System.out.println();
		}

		String modelName = "lmd_fb";
		new SparseMatrix(qData).writeObject(resDir + String.format("%s_%s.ser.gz", "q", modelName));
	}

	private void collectSearchResults(Map<Integer, String> docIdMap, List<BaseQuery> bqs, List<SparseVector> dData,
			CounterMap<String, String> resData) throws Exception {

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector scores = dData.get(i);
			for (int j = 0; j < scores.size(); j++) {
				int dseq = scores.indexAt(j);
				double score = scores.valueAt(j);
				String docid = docIdMap.get(dseq);
				resData.incrementCount(bq.getId(), docid, score);
			}
		}
	}

	private void collectSearchResults(DocumentIdMap dim, List<BaseQuery> bqs, List<SparseVector> dData,
			CounterMap<String, String> resData) throws Exception {

		Map<Integer, String> m = Generics.newHashMap();

		for (int i = 0; i < dData.size(); i++) {
			SparseVector scores = dData.get(i);
			if (scores.size() > top_k) {
				scores = scores.subVector(top_k);
			}

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

	private void collectSearchResults(DocumentSearcher ds, BaseQuery bq, SparseVector scores,
			CounterMap<String, String> resData, int top_k) throws Exception {

		scores = scores.subVector(top_k);
		scores.sortIndexes();

		for (int i = 0; i < scores.size() && i < top_k; i++) {
			int dseq = scores.indexAt(i);
			double score = scores.valueAt(i);
			String docid = ds.getDocumentCollection().get(dseq).getFirst();
			resData.incrementCount(bq.getId(), docid, score);
		}
	}

	public void evaluate() throws Exception {
		CounterMap<String, String> sr1 = FileUtils.readStringCounterMap(MIRPath.TREC_CDS_2014_DIR + "sr_kld.ser.gz");
		CounterMap<String, String> sr2 = FileUtils.readStringCounterMap(MIRPath.TREC_CDS_2014_DIR + "sr_kld-fb.ser.gz");
		CounterMap<String, String> relvData = RelevanceReader
				.readTrecCdsRelevances(MIRPath.TREC_CDS_2014_REL_JUDGE_FILE);

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

	public void formatTrecOutput() throws Exception {
		String[] resFileNames = { "sr_lmd-rf.ser.gz", "sr_lmd-rf_emb.ser.gz", "sr_lmd-rf_cbeem.ser.gz" };
		String[] runNames = { "KISTI01", "KISTI02", "KISTI03" };

		SetMap<String, String> toRemoveData = Generics.newSetMap();

		for (String line : FileUtils.readLinesFromText(dataDir + "t1_list_excluded.txt")) {
			List<String> ps = StrUtils.split(line);
			toRemoveData.put(ps.get(0), ps.get(1));
		}

		for (int i = 0; i < resFileNames.length; i++) {
			String resFileName = resDir + resFileNames[i];
			String runName = runNames[i];
			String trecResFileName = dataDir + "trec_format/" + String.format("%s.txt", runName);
			CounterMap<String, String> sr = FileUtils.readStringCounterMap(resFileName);
			for (String qid : sr.keySet()) {
				Counter<String> scores = sr.getCounter(qid);
				Set<String> toRemove = toRemoveData.get(qid);
				scores.prune(toRemove);
				scores.keepTopNKeys(1000);
			}
			TrecFormatter.format(sr, runName, trecResFileName);
		}
	}

	private Map<String, Integer> getDocumentIdMap() throws Exception {
		Map<String, Integer> ret = Generics.newHashMap();
		for (String line : FileUtils.readLinesFromText(dataDir + "relv_did_dseq.txt")) {
			String[] ps = line.split("\t");
			String did = ps[0];
			int dseq = Integer.parseInt(ps[1]);
			ret.put(did, dseq);
		}
		return ret;
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

	public void runAnalysis() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);
		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());
		List<SparseVector> qData = Generics.newArrayList(bqs.size());
		List<SparseVector> dData = Generics.newArrayList(bqs.size());

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		ds.setTopK(top_k);

		LMScorer scorer = (LMScorer) ds.getScorer();

		Map<String, Integer> docIdMap = Generics.newHashMap();

		for (String line : FileUtils.readLinesFromText(dataDir + "relv_did_dseq.txt")) {
			String[] ps = line.split("\t");
			String did = ps[0];
			int dseq = Integer.parseInt(ps[1]);
			docIdMap.put(did, dseq);
		}

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector Q = ds.index(bq.getSearchText());
			qData.add(Q);
		}

		DenseVector medicalPriors = new DenseVector(dataDir + "doc_prior_medical.ser.gz");
		DenseVector qualityPriors = new DenseVector(dataDir + "doc_prior_quality-2.ser.gz");

		FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(), ds.getWordFilter());

		for (int i = 0; i < qData.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector Q = qData.get(i);
			Counter<String> c = relvData.getCounter(bq.getId());

			SparseVector relvs = new SparseVector(c.size());
			Set<Integer> toKeep = Generics.newHashSet();
			Set<Integer> toRemove = Generics.newHashSet();

			{
				int j = 0;
				for (Entry<String, Double> e : c.entrySet()) {
					String did = e.getKey();
					double relv = e.getValue();
					int dseq = docIdMap.get(did);
					if (dseq < 0) {
						continue;
					}
					relvs.addAt(j++, dseq, relv);

					if (relv > 0) {
						toKeep.add(dseq);
					} else {
						toRemove.add(dseq);
					}
				}
			}
		}
	}

	public void runInitSearch() throws Exception {
		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);
		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());
		List<SparseVector> qData = Generics.newArrayList(bqs.size());
		List<SparseVector> dData = Generics.newArrayList(bqs.size());

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		ds.setTopK(top_k);

		ds.getWordFilter().buildStopIds2();

		// Map<String, Integer> docIdMap = getDocumentIdMap();

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector Q = ds.index(bq.getSearchText());
			qData.add(Q);
		}

		boolean use_lemma_expansion = false;
		boolean use_relevance_feedback = false;

		if (use_lemma_expansion) {
			LemmaExpander le = new LemmaExpander(ds.getWordFilter(),
					FileUtils.readStringHashMapFromText(MIRPath.CLEF_EH_2017_DIR + "lemma.txt"));
			for (int i = 0; i < qData.size(); i++) {
				SparseVector Q = qData.get(i);
				Q = le.expand(Q);
				qData.set(i, Q);
			}
		}

		String modelName = "lmd";

		if (modelName.equals("mrf")) {
			ds.setScorer(new MRFScorer(ds));
			MRFScorer scorer = (MRFScorer) ds.getScorer();
			// scorer.setDocumentPriors(qualityPriors);
			scorer.setPhraseSize(2);
		} else if (modelName.equals("wmrf")) {
			// Counter<String> phrsDocFreqs =
			// FileUtils.readStringCounterFromText("../../data/medical_ir/phrs/phrs_freq.txt");
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
				docNorms = VSMScorer.getDocNorms(ds.getDocumentCollection());
				docNorms.writeObject(normFileName);
			}
			ds.setScorer(new VSMScorer(ds, docNorms));
		}

		{
			List<SparseVector> res = ds.search(qData, null, 2);
			dData.addAll(res);
		}

		Map<Integer, String> docIdMap = Generics.newHashMap();

		for (SparseVector scores : dData) {
			for (int dseq : scores.indexes()) {
				docIdMap.put(dseq, "");
			}
		}

		for (int dseq : docIdMap.keySet()) {
			String did = ds.getDocumentCollection().getDocId(dseq);
			docIdMap.put(dseq, did);
		}

		collectSearchResults(docIdMap, bqs, dData, srData);

		// FileUtils.deleteFilesUnder(resDir);

		Performance p = PerformanceEvaluator.evalute(srData, relvData);
		p.writeObject(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		System.out.println(p);

		FileUtils.writeStringCounterMap(resDir + String.format("%s_%s.ser.gz", "sr", modelName), srData);
		new SparseMatrix(qData).writeObject(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		new SparseMatrix(dData).writeObject(resDir + String.format("%s_%s.ser.gz", "d", modelName));

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
		String fbName = "doc-prior";

		Performance p1 = new Performance(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		SparseMatrix qData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		System.out.println(p1.toString());

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		ds.setUseCache(false);

		DenseVector medicalPriors = new DenseVector(dataDir + "doc_prior_medical.ser.gz");
		DenseVector qualityPriors = new DenseVector(dataDir + "doc_prior_quality-2.ser.gz");
		DenseVector stopRatioPriors = new DenseVector(dataDir + "doc_prior_stop-ratio.ser.gz");

		// QueryModelBuilder qmb = new QueryModelBuilder(new
		// DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR).getVocab(),
		// new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2016_DIR + "emb/glove_ra.ser",
		// true));

		boolean use_qbg = false;
		boolean use_pseudo_relevance_feedback = true;
		boolean use_doc_prior = false;
		boolean use_relevance_feedback = false;
		int top_n = 4000;

		boolean use_bm25 = true;

		if (use_bm25) {
			BM25Scorer scorer = new BM25Scorer(ds);

			for (int i = 0; i < qData.rowSize(); i++) {
				Timer timer = Timer.newTimer();

				SparseVector Q = qData.rowAt(i);

				SparseVector scores = dData.rowAt(i);
				scores = scores.subVector(top_n);
				SparseVector scores2 = scorer.scoreFromCollection(Q, scores);
				scorer.postprocess(scores2);

				scores2 = VectorMath.addAfterMultiply(new SparseVector[] { scores, scores2 },
						new double[] { 0.5, 0.5 });
				scorer.postprocess(scores2);

				dData.set(i, scores2);
				qData.set(i, Q);

				System.out.printf("No:\t%d\n", i);
				System.out.println(VectorUtils.toCounter(Q, ds.getVocab()));
				System.out.println(timer.stop() + "\n");
			}
		}

		if (use_doc_prior) {
			double mixture_medical = medicalPriors.sum() / qualityPriors.sum();

			for (int i = 0; i < dData.rowSize(); i++) {
				SparseVector scores = dData.rowAt(i);
				double[] mixtures = { 0.7, 0.1, 0.2 };
				ArrayMath.normalize(mixtures);

				for (int j = 0; j < scores.size(); j++) {
					int dseq = scores.indexAt(j);
					double score = scores.valueAt(j);
					double prior1 = medicalPriors.value(dseq);
					double prior2 = qualityPriors.value(dseq);
					double prior3 = stopRatioPriors.value(dseq);
					// double score2 = ArrayMath.dotProduct(mixtures, new double[] { score, prior1,
					// prior2 });

					// double score2 = score * (mixture_medical * prior1 + (1 - mixture_medical) *
					// prior2);
					double score2 = score * prior3;
					scores.setAt(j, dseq, score2);
				}
				scores.summation();
				scores.sortValues();
				dData.set(i, scores);
			}
		}

		if (use_relevance_feedback) {
			LMScorer scorer = (LMScorer) ds.getScorer();

			FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(),
					ds.getWordFilter());
			SparseMatrix rfbData = new SparseMatrix(resDir + "rfb_lmd_doc-prior.ser.gz");

			for (int i = 0; i < qData.rowSize(); i++) {
				Timer timer = Timer.newTimer();

				SparseVector Q = qData.rowAt(i);
				SparseVector lm_q1 = Q.copy();
				lm_q1.normalize();

				SparseVector lm_fb = rfbData.rowAt(i);

				SparseVector lm_q2 = fb.updateQueryModel(lm_q1, lm_fb);

				SparseVector scores = dData.rowAt(i);
				// scores = scores.subVector(top_n);
				SparseVector scores2 = scorer.scoreFromCollection(lm_q2, scores);
				scorer.postprocess(scores2);

				dData.set(i, scores2);
				qData.set(i, Q);

				System.out.printf("No:\t%d\n", i);
				System.out.println(VectorUtils.toCounter(Q, ds.getVocab()));
				System.out.println(timer.stop() + "\n");
			}
		}

		if (use_pseudo_relevance_feedback) {
			LMScorer scorer = (LMScorer) ds.getScorer();

			FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(),
					ds.getWordFilter());
			fb.setFeedbackMixture(0.1);
			fb.setFbDocSize(20);

			for (int i = 0; i < qData.rowSize(); i++) {
				Timer timer = Timer.newTimer();

				SparseVector Q = ds.index(bqs.get(i).getSearchText());
				SparseVector lm_q1 = VectorUtils.toSparseVector(Q);
				lm_q1.normalize();

				SparseVector scores = dData.rowAt(i);
				scores = scores.subVector(top_n);

				SparseVector lm_fb = new SparseVector(0);
				SparseVector lm_q2 = lm_q1;

				lm_fb = fb.buildRM1(scores, 0, qualityPriors);
				lm_q2 = fb.updateQueryModel(lm_q1, lm_fb);

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
		}

		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());

		collectSearchResults(new DocumentIdMap(idxDir), bqs, dData, srData);

		String pFileName = resDir + String.format("%s_%s_%s.ser.gz", "p", modelName, fbName);
		String srFileName = resDir + String.format("%s_%s_%s.ser.gz", "sr", modelName, fbName);
		String qFileName = resDir + String.format("%s_%s_%s.ser.gz", "q", modelName, fbName);
		String dFileName = resDir + String.format("%s_%s_%s.ser.gz", "d", modelName, fbName);

		Performance p2 = PerformanceEvaluator.evalute(srData, relvData);
		PerformanceComparator.compare(p1, p2);
		p2.writeObject(pFileName);

		FileUtils.writeStringCounterMap(srFileName, srData);

		new SparseMatrix(qData).writeObject(qFileName);
		new SparseMatrix(dData).writeObject(dFileName);

		System.out.println(p2);
	}

	public void runReranking2() throws Exception {
		System.out.println(resDir);

		String modelName = "lmd-rf";
		String fbName = "emb";

		Performance p1 = new Performance(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		SparseMatrix qData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		System.out.println(p1.toString());

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);
		ds.setUseCache(false);

		WordSearcher ws = new WordSearcher(DocumentCollection.readVocab(MIRPath.TREC_CDS_2016_COL_DC_DIR + "vocab.ser"),
				new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2016_DIR + "emb/glove_ra.ser", true), null);

		QueryModelBuilder qmb = new QueryModelBuilder(ws.getVocab(), ws.getEmbeddingMatrix());

		DenseVector e_med = new DenseVector(ws.getEmbeddingMatrix().colSize());
		double idf_med = 0;

		int top_n = 4000;

		FeedbackBuilder fb = new FeedbackBuilder(ds.getDocumentCollection(), ds.getInvertedIndex(), ds.getWordFilter());
		fb.setMixtureJM(0.5);

		LMScorer scorer = (LMScorer) ds.getScorer();
		Vocab vocab = ds.getVocab();

		for (int i = 0; i < qData.rowSize(); i++) {
			Timer timer = Timer.newTimer();

			SparseVector Q1 = qData.rowAt(i);
			Counter<String> c1 = VectorUtils.toCounter(Q1, vocab);
			Counter<String> c2 = qmb.build1(c1, e_med, idf_med);
			SparseVector Q2 = VectorUtils.toSparseVector(c2, vocab);

			Q2 = fb.updateQueryModel(Q1, Q2);

			SparseVector scores = dData.rowAt(i);
			scores = scores.subVector(top_n);
			SparseVector scores2 = scorer.scoreFromCollection(Q2, scores);
			scorer.postprocess(scores2);

			dData.set(i, scores2);
			qData.set(i, Q2);

			System.out.printf("No:\t%d\n", i);
			System.out.println(VectorUtils.toCounter(Q1, ds.getVocab()));
			System.out.println(VectorUtils.toCounter(Q2, ds.getVocab()));
			System.out.println(timer.stop() + "\n");
		}

		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());

		collectSearchResults(new DocumentIdMap(idxDir), bqs, dData, srData);

		String pFileName = resDir + String.format("%s_%s_%s.ser.gz", "p", modelName, fbName);
		String srFileName = resDir + String.format("%s_%s_%s.ser.gz", "sr", modelName, fbName);
		String qFileName = resDir + String.format("%s_%s_%s.ser.gz", "q", modelName, fbName);
		String dFileName = resDir + String.format("%s_%s_%s.ser.gz", "d", modelName, fbName);

		Performance p2 = PerformanceEvaluator.evalute(srData, relvData);
		PerformanceComparator.compare(p1, p2);
		p2.writeObject(pFileName);

		FileUtils.writeStringCounterMap(srFileName, srData);

		new SparseMatrix(qData).writeObject(qFileName);
		new SparseMatrix(dData).writeObject(dFileName);

		System.out.println(p2);
	}

	public void runReranking3() throws Exception {
		System.out.println(resDir);

		String modelName = "lmd-rf";
		String fbName = "cbeem";

		Performance p1 = new Performance(resDir + String.format("%s_%s.ser.gz", "p", modelName));
		SparseMatrix qData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		System.out.println(p1.toString());

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds1 = new DocumentSearcher(idxDir, stopwordFileName);
		ds1.setUseCache(false);

		DocumentSearcher ds2 = new DocumentSearcher(MIRPath.TREC_CDS_2016_COL_DC_DIR, stopwordFileName);
		ds2.setTopK(top_k);

		FeedbackBuilder fb1 = new FeedbackBuilder(ds1.getDocumentCollection(), ds1.getInvertedIndex(),
				ds1.getWordFilter());
		FeedbackBuilder fb2 = new FeedbackBuilder(ds2.getDocumentCollection(), ds2.getInvertedIndex(),
				ds2.getWordFilter());

		List<FeedbackBuilder> fbs = Generics.newArrayList();
		fbs.add(fb1);
		fbs.add(fb2);

		List<DocumentSearcher> dss = Generics.newArrayList();
		dss.add(ds1);
		dss.add(ds2);

		LMScorer scorer = (LMScorer) ds1.getScorer();

		DenseVector colWeights = new DenseVector(new double[] { 0.8, 0.2 });
		double mixture_col_weight = 0;
		int top_n = 2000;

		for (int i = 0; i < qData.rowSize(); i++) {
			Timer timer = Timer.newTimer();

			SparseVector lm_q1 = qData.rowAt(i);
			SparseVector lm_q2 = VectorUtils.toSparseVector(lm_q1, ds1.getVocab(), ds2.getVocab());

			SparseVector scores1 = dData.rowAt(i);
			SparseVector scores2 = ds2.search(lm_q2);

			List<SparseVector> scoreData = Generics.newArrayList();
			scoreData.add(scores1);
			scoreData.add(scores2);

			SparseVector lm_fb = null;

			if (fbName.equals("eem")) {
				lm_fb = fb1.buildEEM(fbs, scoreData, null, colWeights, mixture_col_weight);
			} else if (fbName.equalsIgnoreCase("cbeem")) {
				lm_fb = fb1.buildCBEEM(dss, scoreData, null, colWeights, mixture_col_weight);
			}

			SparseVector lm_q3 = fb1.updateQueryModel(lm_q1, lm_fb);

			scores1 = scores1.subVector(top_n);

			SparseVector scores3 = scorer.scoreFromCollection(lm_q3, scores1);
			scorer.postprocess(scores3);

			dData.set(i, scores3);
			qData.set(i, lm_q3);

			System.out.printf("No:\t%d\n", i);
			System.out.println(VectorUtils.toCounter(lm_q1, ds1.getVocab()));
			System.out.println(VectorUtils.toCounter(lm_q3, ds1.getVocab()));
			System.out.println(timer.stop() + "\n");
		}

		CounterMap<String, String> srData = Generics.newCounterMap(bqs.size());

		collectSearchResults(new DocumentIdMap(idxDir), bqs, dData, srData);

		String pFileName = resDir + String.format("%s_%s_%s.ser.gz", "p", modelName, fbName);
		String srFileName = resDir + String.format("%s_%s_%s.ser.gz", "sr", modelName, fbName);
		String qFileName = resDir + String.format("%s_%s_%s.ser.gz", "q", modelName, fbName);
		String dFileName = resDir + String.format("%s_%s_%s.ser.gz", "d", modelName, fbName);

		Performance p2 = PerformanceEvaluator.evalute(srData, relvData);
		PerformanceComparator.compare(p1, p2);
		p2.writeObject(pFileName);

		FileUtils.writeStringCounterMap(srFileName, srData);

		new SparseMatrix(qData).writeObject(qFileName);
		new SparseMatrix(dData).writeObject(dFileName);

		System.out.println(p2);
	}

	public void test() {
		String[] daraDirs = { MIRPath.BIOASQ_COL_DC_DIR, MIRPath.TREC_CDS_2016_COL_DC_DIR };
	}

}
