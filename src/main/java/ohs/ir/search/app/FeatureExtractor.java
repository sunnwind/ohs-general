package ohs.ir.search.app;

import java.util.List;

import ohs.corpus.type.EnglishNormalizer;
import ohs.corpus.type.EnglishTokenizer;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.general.NLPUtils;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.medical.query.QueryReader;
import ohs.ir.medical.query.RelevanceReader;
import ohs.ir.medical.query.TrecCdsQuery;
import ohs.ir.search.model.LMScorer;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.BidMap;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class FeatureExtractor {

	public static void build1() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser");

		FeatureExtractor featExt = new FeatureExtractor(ds, E);

		EnglishTokenizer et = new EnglishTokenizer();

		int top_k = 1000;

		for (int u = 0; u < 2; u++) {
			String qFileName = MIRPath.TREC_CDS_2014_QUERY_FILE;
			String relFileName = MIRPath.TREC_CDS_2014_REL_JUDGE_FILE;
			String labelFileName = MIRPath.TREC_CDS_2014_DIR + "l2r_2014_label.ser.gz";
			String dataFileName = MIRPath.TREC_CDS_2014_DIR + "l2r_2014_data.ser.gz";

			if (u == 1) {
				qFileName = MIRPath.TREC_CDS_2015_QUERY_A_FILE;
				relFileName = MIRPath.TREC_CDS_2015_REL_JUDGE_FILE;
				labelFileName = MIRPath.TREC_CDS_2014_DIR + "l2r_2015_label.ser.gz";
				dataFileName = MIRPath.TREC_CDS_2014_DIR + "l2r_2015_data.ser.gz";
			}

			List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(qFileName);

			CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

			DenseMatrix data = new DenseMatrix();
			IntegerArray labels = new IntegerArray();

			for (int i = 0; i < bqs.size(); i++) {
				BaseQuery bq = bqs.get(i);
				TrecCdsQuery tcq = (TrecCdsQuery) bq;

				Counter<String> docRels = relvData.getCounter(tcq.getId());

				System.out.println(bq);

				String st = StrUtils.join(" ", et.tokenize(bq.getSearchText()));

				SparseVector Q = ds.index(st);
				SparseVector lm_q = Q.copy();
				lm_q.normalize();

				SparseVector docScores = ds.search(lm_q);

				docScores.keepTopN(top_k);
				docScores.sortValues();

				for (int j = 0; j < docScores.size(); j++) {
					int docseq = docScores.indexAt(j);
					String docid = ds.getDocumentCollection().getSents(docseq).getFirst();
					double relevance = docRels.getCount(docid);
					DoubleArray feats = featExt.extract(Q, docseq);
					data.add(new DenseVector(feats.values()));
					labels.add((int) relevance);
				}
			}

			labels.writeObject(labelFileName);
			data.writeObject(dataFileName);
		}

	}

	public static void build2() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser");

		BidMap<String, Integer> idToSeq = Generics.newBidMap();

		for (String line : FileUtils.readLinesFromText(MIRPath.TREC_CDS_2014_DIR + "id_to_docseq.txt")) {
			String[] parts = line.split("\t");
			idToSeq.put(parts[0], Integer.parseInt(parts[1]));
		}

		FeatureExtractor ext = new FeatureExtractor(ds, E);

		EnglishTokenizer et = new EnglishTokenizer();

		for (int u = 0; u < 2; u++) {
			String qFileName = MIRPath.TREC_CDS_2014_QUERY_FILE;
			String relFileName = MIRPath.TREC_CDS_2014_REL_JUDGE_FILE;
			String labelFileName = MIRPath.TREC_CDS_2014_DIR + "l2r_2014_label.ser.gz";
			String dataFileName = MIRPath.TREC_CDS_2014_DIR + "l2r_2014_data.ser.gz";

			if (u == 1) {
				qFileName = MIRPath.TREC_CDS_2015_QUERY_A_FILE;
				relFileName = MIRPath.TREC_CDS_2015_REL_JUDGE_FILE;
				labelFileName = MIRPath.TREC_CDS_2014_DIR + "l2r_2015_label.ser.gz";
				dataFileName = MIRPath.TREC_CDS_2014_DIR + "l2r_2015_data.ser.gz";
			}

			List<BaseQuery> bqs = QueryReader.readTrecCdsQueries(qFileName);

			CounterMap<String, String> relvData = RelevanceReader.readTrecCdsRelevances(relFileName);

			DenseMatrix data = new DenseMatrix();
			IntegerArray labels = new IntegerArray();

			for (int i = 0; i < bqs.size(); i++) {
				BaseQuery bq = bqs.get(i);
				TrecCdsQuery tcq = (TrecCdsQuery) bq;

				Counter<String> docRels = relvData.getCounter(tcq.getId());

				String st = StrUtils.join(" ", et.tokenize(bq.getSearchText()));

				SparseVector Q = ds.index(st);

				List<String> docids = docRels.getSortedKeys();
				int relevant = 0;
				for (String docid : docids) {
					double relevance = docRels.getCount(docid);
					Integer docseq = idToSeq.getValue(docid);

					if (docseq == null) {
						continue;
					}

					if (relevance == 0) {
						break;
					}
					relevant++;
				}

				int size = relevant * 3;

				for (int j = 0; j < size && j < docids.size(); j++) {
					String docid = docids.get(j);
					double relevance = docRels.getCount(docid);
					Integer docseq = idToSeq.getValue(docid);

					if (docseq == null) {
						continue;
					}
					DoubleArray feats = ext.extract(Q, docseq);
					data.add(new DenseVector(feats.values()));
					labels.add((int) relevance);
				}
			}

			labels.writeObject(labelFileName);
			data.writeObject(dataFileName);

			System.out.println(data.size());
		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		build2();

		System.out.println("process ends.");
	}

	private DocumentSearcher ds;

	private RandomAccessDenseMatrix E;

	public FeatureExtractor(DocumentSearcher ds, RandomAccessDenseMatrix E) {
		this.ds = ds;
		this.E = E;
	}

	// public DenseMatrix extract(SparseVector Q, SparseVector docScores) throws Exception {
	// DenseMatrix ret = new DenseMatrix(docScores.size());
	// for (int i = 0; i < docScores.size(); i++) {
	// int docseq = docScores.indexAt(i);
	// double score = docScores.valueAt(i);
	// DoubleArray feats = extract(Q, docseq);
	// feats.add(score);
	// feats.trimToSize();
	// ret.add(new DenseVector(feats.elementData()));
	// }
	// return ret;
	// }

	public DoubleArray extract(SparseVector Q, int docseq) throws Exception {
		Q = VectorUtils.toSparseVector(Q);

		SparseVector d = ds.getDocumentCollection().getDocVector(docseq);
		double num_docs = ds.getVocab().getDocCnt();

		DoubleArray feats = new DoubleArray();

		{

			SparseVector idfs_d = new SparseVector(d.size());
			SparseVector tfidfs_d = new SparseVector(d.size());

			for (int i = 0; i < d.size(); i++) {
				int w = d.indexAt(i);
				double cnt = d.valueAt(i);
				double doc_freq = ds.getVocab().getDocFreq(w);
				double idf = TermWeighting.idf(num_docs, doc_freq);
				double tfidf = TermWeighting.tfidf(cnt, num_docs, doc_freq);
				idfs_d.addAt(i, w, idf);
				tfidfs_d.addAt(i, w, tfidf);
			}

			SparseVector cnts_qd = new SparseVector(Q.size());
			SparseVector idfs_qd = new SparseVector(Q.size());
			SparseVector tfidfs_qd = new SparseVector(Q.size());

			for (int i = 0; i < Q.size(); i++) {
				int w = Q.indexAt(i);
				cnts_qd.addAt(i, w, d.value(w));
				idfs_qd.addAt(i, w, idfs_d.value(w));
				tfidfs_qd.addAt(i, w, tfidfs_d.value(w));
			}

			feats.add(Q.sum());
			feats.add(d.sum());
			feats.add(cnts_qd.sum());
			feats.add(cnts_qd.sum() / d.sum());

			feats.add(idfs_qd.sum());
			feats.add(idfs_qd.sum() / idfs_d.sum());

			feats.add(tfidfs_qd.sum());
			feats.add(tfidfs_qd.sum() / tfidfs_d.sum());

			feats.add(VectorMath.cosine(Q, d));

			SparseVector lm_q = Q.copy();
			lm_q.normalize();

			feats.add(LMScorer.score(lm_q, d, ds.getVocab(), 2000, 0));
			feats.add(LMScorer.score(lm_q, d, ds.getVocab(), 0, 0.5));
			feats.add(LMScorer.score(lm_q, d, ds.getVocab(), 2000, 0.5));

		}
		feats.trimToSize();
		return feats;
	}

}
