package ohs.ir.search.app;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import it.unimi.dsi.fastutil.doubles.Double2IntOpenCustomHashMap;
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
import ohs.ir.search.model.BM25Scorer;
import ohs.ir.search.model.FeedbackBuilder;
import ohs.ir.search.model.LMScorer;
import ohs.ir.search.model.MRFScorer;
import ohs.ir.search.model.ParsimoniousLanguageModelEstimator;
import ohs.ir.search.model.VSMScorer;
import ohs.ir.search.model.WeightedMRFScorer;
import ohs.ir.search.model.WordProximities;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class TrecPMExperiments {

	public static int COR_TYPE = 1;

	public static String dataDir = MIRPath.TREC_PM_2017_DIR;

	public static String idxDir = MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR;

	public static String queryFileName = MIRPath.TREC_PM_2017_QUERY_FILE;

	public static String relFileName = "";

	public static String resDir = dataDir + "res/medline/";

	public static String stopwordFileName = MIRPath.STOPWORD_INQUERY_FILE;

	// static {
	//
	// if (COR_TYPE == 0) {
	// dataDir = MIRPath.TREC_CDS_2016_DIR;
	// idxDir = MIRPath.TREC_CDS_2016_COL_DC_DIR;
	// queryFileName = MIRPath.TREC_CDS_2016_QUERY_FILE;
	// relFileName = MIRPath.TREC_CDS_2016_REL_JUDGE_FILE;
	// } else if (COR_TYPE == 1) {
	// dataDir = MIRPath.TREC_PM_2017_DIR;
	// idxDir = MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR;
	// queryFileName = MIRPath.TREC_PM_2017_QUERY_FILE;
	// relFileName = "";
	// resDir = dataDir + "res/medline/";
	// } else if (COR_TYPE == 2) {
	// dataDir = MIRPath.TREC_PM_2017_DIR;
	// idxDir = MIRPath.TREC_PM_2017_COL_CLINICAL_DC_DIR;
	// queryFileName = MIRPath.TREC_PM_2017_QUERY_FILE;
	// relFileName = "";
	// resDir = dataDir + "res/clinical/";
	// }
	// }

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		TrecPMExperiments e = new TrecPMExperiments();

		int[] corTypes = { 0, 1, 2 };

		for (int i = 0; i < corTypes.length; i++) {
			int corType = corTypes[i];
			if (corType < 1) {
				continue;
			}

			if (corType == 0) {
				dataDir = MIRPath.TREC_CDS_2016_DIR;
				idxDir = MIRPath.TREC_CDS_2016_COL_DC_DIR;
				queryFileName = MIRPath.TREC_CDS_2016_QUERY_FILE;
				relFileName = MIRPath.TREC_CDS_2016_REL_JUDGE_FILE;
			} else if (corType == 1) {
				dataDir = MIRPath.TREC_PM_2017_DIR;
				idxDir = MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR;
				queryFileName = MIRPath.TREC_PM_2017_QUERY_FILE;
				relFileName = "";
				resDir = dataDir + "res/medline/";
			} else if (corType == 2) {
				dataDir = MIRPath.TREC_PM_2017_DIR;
				idxDir = MIRPath.TREC_PM_2017_COL_CLINICAL_DC_DIR;
				queryFileName = MIRPath.TREC_PM_2017_QUERY_FILE;
				relFileName = "";
				resDir = dataDir + "res/clinical/";
			}

			// e.runInitSearch();
			// e.runReranking();
			// e.runReranking2();
			// e.runReranking3();
			// e.runReranking4();
			e.formatTrecOutput();
		}

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
		String[] resFileNames = { "d_lmd", "d_lmd_rm3", "d_lmd_fb-rw", "d_lmd_fb-eem", "d_lmd_fb-rw-eem-mix" };
		String[] runNames = { "KISTI01", "KISTI02", "KISTI03", "KISTI04", "KISTI05" };

		if (resDir.contains("clinical")) {
			for (int i = 0; i < runNames.length; i++) {
				runNames[i] = runNames[i] + "CT";
			}
		}

		for (int i = 0; i < resFileNames.length; i++) {
			DocumentIdMap dim = new DocumentIdMap(idxDir);

			String resFileName = resDir + resFileNames[i] + ".ser.gz";
			String runName = runNames[i];
			String trecResFileName = resDir.replace("res", "trec_format") + String.format("%s.txt", runName);

			SparseMatrix dData = new SparseMatrix(resFileName);

			CounterMap<String, String> sr = Generics.newCounterMap();

			for (int j = 0; j < dData.rowSize(); j++) {
				SparseVector scores = dData.rowAt(j);
				Counter<String> c = sr.getCounter(new DecimalFormat("00").format(j + 1) + "");

				for (int k = 0; k < scores.size(); k++) {
					int dseq = scores.indexAt(k);
					double score = scores.valueAt(k);
					String did = dim.get(dseq);
					c.setCount(did, score);
				}

				c.keepTopNKeys(1000);

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

		System.out.println(ds.getDocumentCollection().toString());

		ds.getWordFilter().buildStopIds2();

		// Map<String, Integer> docIdMap = getDocumentIdMap();

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector Q = ds.index(bq.getSearchText());
			qData.add(Q);
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
		String fbName = "rm3";

		SparseMatrix qData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "q", modelName));
		SparseMatrix dData = new SparseMatrix(resDir + String.format("%s_%s.ser.gz", "d", modelName));

		List<BaseQuery> bqs = QueryReader.readQueries(queryFileName);
		CounterMap<String, String> relvData = RelevanceReader.readRelevances(relFileName);

		DocumentSearcher ds = new DocumentSearcher(idxDir, stopwordFileName);

		ds.getWordFilter().buildStopIds2();

		LMScorer scorer = (LMScorer) ds.getScorer();

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

		LMScorer scorer = (LMScorer) ds.getScorer();

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

		LMScorer scorer = (LMScorer) ds1.getScorer();

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

		LMScorer scorer = (LMScorer) ds1.getScorer();

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

	public void test() throws Exception {
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

		Counter<String> phrsWeights = FileUtils.readStringCounterFromText(MIRPath.DATA_DIR + "phrs/phrs_weight.txt");

		PhraseMapper<Integer> pm = new PhraseMapper<>(PhraseMapper.createTrie(phrsWeights.keySet(), ds.getVocab()));

		TextFileWriter writer = new TextFileWriter(dataDir + "q_docs.txt");

		for (int i = 0; i < bqs.size(); i++) {
			BaseQuery bq = bqs.get(i);
			SparseVector q = qData.rowAt(i);
			SparseVector scores = dData.rowAt(i);
			StringBuffer sb = new StringBuffer();
			sb.append(bq.toString());

			SparseVector lm_q = qData.rowAt(i);

			for (int j = 0; j < scores.size() && j < 10; j++) {
				int dseq = scores.indexAt(j);
				double score = scores.valueAt(j);
				Pair<String, String> p = ds.getDocumentCollection().getText(dseq);

				SparseVector dv = ds.getDocumentCollection().getDocVector(dseq);

				Counter<String> c = Generics.newCounter();

				for (int w : lm_q.indexes()) {
					double cnt = dv.value(w);
					if (cnt > 0) {
						c.setCount(ds.getVocab().getObject(w), cnt);
					}
				}

				IntegerArray d = ds.getDocumentCollection().get(dseq).getSecond();
				Counter<String> cc = Generics.newCounter();

				{

					Counter<String> wordCnts = VectorUtils.toCounter(dv, ds.getVocab());
					Indexer<String> wordIdxer = Generics.newIndexer(wordCnts.keySet());

					int window_size = 5;
					CounterMap<String, String> cm = Generics.newCounterMap();

					for (IntegerArray s : DocumentCollection.toMultiSentences(d)) {
						List<String> ws = Arrays.asList(ds.getVocab().getObjects(s));

						for (int m = 0; m < ws.size(); m++) {
							String w1 = ws.get(m);

							for (int n = m + 1; n < Math.min(ws.size(), m + window_size); n++) {
								String w2 = ws.get(n);
								double dist = n - m;
								double sim = 1d / dist;
								cm.incrementCount(w1, w2, sim);
								cm.incrementCount(w2, w1, sim);
							}
						}
					}

					DenseMatrix T = VectorUtils.toDenseMatrix(cm, wordIdxer, wordIdxer, false);

					SparseVector tfidfs = VectorUtils.toSparseVector(wordCnts, wordIdxer);

					for (int m = 0; m < tfidfs.size(); m++) {
						int w = tfidfs.indexAt(m);
						double cnt = tfidfs.valueAt(m);
						String word = wordIdxer.getObject(w);
						double tfidf = TermWeighting.tfidf(cnt, ds.getVocab().getDocCnt(),
								ds.getVocab().getDocFreq(word));
						tfidfs.setAt(m, tfidf);
					}
					tfidfs.summation();

					for (int w1 = 0; w1 < T.rowSize(); w1++) {
						double weight1 = tfidfs.value(w1);
						DenseVector t = T.row(w1);

						for (int w2 = 0; w2 < t.size(); w2++) {
							double weight2 = tfidfs.value(w2);
							double sim = t.valueAt(w2);
							double new_sim = sim * weight1 * weight2;
							t.setAt(w2, new_sim);
						}
					}

					T.sumRows();
					T.normalizeColumns();

					DenseVector C = new DenseVector(dv.size());

					DenseVector B = new DenseVector(dv.size());

					for (String word : ds.getVocab().getObjects(q.indexes())) {
						int w = wordIdxer.indexOf(word);
						if (w < 0) {
							continue;
						}
						B.add(w, 1);
					}
					B.normalize();

					ArrayMath.randomWalk(T.values(), C.values(), B.values(), 100);

					cc = VectorUtils.toCounter(C, wordIdxer);
				}

				Counter<String> tmp1 = Generics.newCounter();
				Counter<String> tmp2 = Generics.newCounter();

				for (IntPair r : pm.map(d.toArrayList())) {
					String phrs = StrUtils.join(" ", ds.getVocab().getObjects(d.subArray(r.getFirst(), r.getSecond())));
					double weight = phrsWeights.getCount(phrs);
					tmp1.incrementCount(phrs, 1);
					tmp2.incrementCount(phrs, weight);
				}

				ParsimoniousLanguageModelEstimator plm = new ParsimoniousLanguageModelEstimator(
						ds.getDocumentCollection());
				SparseVector lm_d = plm.estimate(dv);

				Counter<String> cc3 = VectorUtils.toCounter(lm_d, ds.getVocab());

				sb.append(String.format("\ndseq:\t%d\nscore:\t%f", dseq, score));
				sb.append(String.format("\ndid:\t%s\ntext:\n%s", p.getFirst(), p.getSecond()));
				sb.append(String.format("\nqv:\t%s", c.toString(c.size())));
				sb.append(String.format("\nphrs:\n%s", tmp2.toStringSortedByValues(true, true, tmp2.size(), "\t")));
				sb.append(String.format("\n%s", tmp2.toString(tmp2.size())));
				sb.append(String.format("\n%s", cc.toString(cc.size())));
				sb.append(String.format("\n%s", cc3.toString(cc3.size())));
				sb.append("\n\n");
			}

			writer.write(sb.toString() + "\n\n");

		}
		writer.close();

	}

}
