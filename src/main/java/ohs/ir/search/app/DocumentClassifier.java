package ohs.ir.search.app;

import java.util.List;

import ohs.corpus.type.EnglishTokenizer;
import ohs.corpus.type.StringTokenizer;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.medical.query.QueryReader;
import ohs.ir.medical.query.RelevanceReader;
import ohs.ir.medical.query.TrecCdsQuery;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.layer.BatchNormalization;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.types.generic.BidMap;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class DocumentClassifier {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// generate();
		generate2();
		// train();

		System.out.println("process ends.");
	}

	public static void generate() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		RandomAccessDenseMatrix radm = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser");

		List<BaseQuery> bqs1 = QueryReader.readTrecCdsQueries(MIRPath.TREC_CDS_2014_QUERY_FILE);
		List<BaseQuery> bqs2 = QueryReader.readTrecCdsQueries(MIRPath.TREC_CDS_2015_QUERY_A_FILE);

		CounterMap<String, String> relvData1 = RelevanceReader
				.readTrecCdsRelevances(MIRPath.TREC_CDS_2014_REL_JUDGE_FILE);
		CounterMap<String, String> relvData2 = RelevanceReader
				.readTrecCdsRelevances(MIRPath.TREC_CDS_2015_REL_JUDGE_FILE);

		BidMap<String, Integer> idTodseq = Generics.newBidMap();

		for (String line : FileUtils.readLinesFromText(MIRPath.TREC_CDS_2014_DIR + "id_to_dseq.txt")) {
			String[] parts = line.split("\t");
			idTodseq.put(parts[0], Integer.parseInt(parts[1]));
		}

		StringTokenizer st = new EnglishTokenizer();

		for (int u = 0; u < 2; u++) {
			IntegerArray labels = new IntegerArray();
			DenseMatrix data = new DenseMatrix();

			CounterMap<String, String> relvData = null;
			List<BaseQuery> bqs = null;

			if (u == 0) {
				relvData = relvData1;
				bqs = bqs1;
			} else {
				relvData = relvData2;
				bqs = bqs2;
			}

			for (int i = 0; i < bqs.size(); i++) {
				TrecCdsQuery tcq = (TrecCdsQuery) bqs.get(i);

				String tmp = StrUtils.join(" ", st.tokenize(tcq.getSearchText()));

				SparseVector Q = ds.index(tmp);

				DenseVector e_q = new DenseVector(radm.colSize());

				for (int w : Q.indexes()) {
					VectorMath.add(radm.row(w), e_q);
				}

				e_q.multiply(1f / Q.size());

				Counter<String> docRels = relvData.getCounter(tcq.getId());

				List<String> docids = docRels.getSortedKeys();
				int relevant_cnt = 0;

				for (int j = 0; j < docids.size(); j++) {
					String docid = docids.get(j);
					Integer dseq = idTodseq.getValue(docid);

					if (dseq == null) {
						continue;
					}

					double relevance = docRels.getCount(docid);
					if (relevance == 0) {
						break;
					}
					relevant_cnt++;
				}

				for (int j = 0; j < Math.min(relevant_cnt * 2, docRels.size()); j++) {
					String docid = docids.get(j);
					Integer dseq = idTodseq.getValue(docid);

					if (dseq == null) {
						continue;
					}

					double relevance = docRels.getCount(docid);

					SparseVector dv = ds.getDocumentCollection().getDocVector(dseq);

					DenseVector e_d = e_q.copy(true);

					for (int w : dv.indexes()) {
						VectorMath.add(radm.row(w), e_d);
					}

					e_d.multiply(1f / dv.size());

					VectorMath.subtract(e_d, e_q, e_d);

					data.add(e_d);
					labels.add((int) relevance);
				}
			}

			String dataFileName = "l2r_2014_data.ser.gz";
			String labelFileName = "l2r_2014_label.ser.gz";

			if (u == 0) {

			} else {
				dataFileName = "l2r_2015_data.ser.gz";
				labelFileName = "l2r_2015_label.ser.gz";
			}

			labels.writeObject(MIRPath.TREC_CDS_2014_DIR + labelFileName);
			data.writeObject(MIRPath.TREC_CDS_2014_DIR + dataFileName);
		}
	}

	public static void generate2() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		RandomAccessDenseMatrix radm = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser");

		List<BaseQuery> bqs1 = QueryReader.readTrecCdsQueries(MIRPath.TREC_CDS_2014_QUERY_FILE);
		List<BaseQuery> bqs2 = QueryReader.readTrecCdsQueries(MIRPath.TREC_CDS_2015_QUERY_A_FILE);

		CounterMap<String, String> relvData1 = RelevanceReader
				.readTrecCdsRelevances(MIRPath.TREC_CDS_2014_REL_JUDGE_FILE);
		CounterMap<String, String> relvData2 = RelevanceReader
				.readTrecCdsRelevances(MIRPath.TREC_CDS_2015_REL_JUDGE_FILE);

		System.out.println(relvData1.totalSize());
		System.out.println(relvData2.totalSize());

		BidMap<String, Integer> idTodseq = Generics.newBidMap();

		for (String line : FileUtils.readLinesFromText(MIRPath.TREC_CDS_2014_DIR + "id_to_dseq.txt")) {
			String[] parts = line.split("\t");
			idTodseq.put(parts[0], Integer.parseInt(parts[1]));
		}

		StringTokenizer st = new EnglishTokenizer();

		Indexer<String> labelIndexer = Generics.newIndexer();
		// labelIndexer.add("None");

		for (int u = 0; u < 2; u++) {
			IntegerArray labels = new IntegerArray();
			DenseMatrix data = new DenseMatrix();

			CounterMap<String, String> relvData = null;
			List<BaseQuery> bqs = null;

			if (u == 0) {
				relvData = relvData1;
				bqs = bqs1;
			} else {
				relvData = relvData2;
				bqs = bqs2;
			}

			for (int i = 0; i < bqs.size(); i++) {
				TrecCdsQuery tcq = (TrecCdsQuery) bqs.get(i);

				String tmp = StrUtils.join(" ", st.tokenize(tcq.getSearchText(false)));

				Counter<String> docRels = relvData.getCounter(tcq.getId());

				List<String> docids = docRels.getSortedKeys();
				int relevant_cnt = 0;

				for (int j = 0; j < docids.size(); j++) {
					String docid = docids.get(j);
					Integer dseq = idTodseq.getValue(docid);
					if (dseq == null) {
						continue;
					}
					relevant_cnt++;
				}

				for (int j = 0; j < Math.min(relevant_cnt * 2, docRels.size()); j++) {
					String docid = docids.get(j);
					Integer dseq = idTodseq.getValue(docid);
					double relevance = docRels.getCount(docid);

					if (dseq == null || relevance == 0) {
						continue;
					}

					SparseVector dv = ds.getDocumentCollection().getDocVector(dseq);

					DenseVector e_d = new DenseVector(radm.colSize());

					for (int w : dv.indexes()) {
						VectorMath.add(radm.row(w), e_d);
					}

					e_d.multiply(1f / dv.size());

					int label = 0;

					label = labelIndexer.getIndex(tcq.getType());

					data.add(e_d);
					labels.add(label);
				}
			}

			String dataFileName = "l2r_2014_data.ser.gz";
			String labelFileName = "l2r_2014_label.ser.gz";

			if (u == 0) {

			} else {
				dataFileName = "l2r_2015_data.ser.gz";
				labelFileName = "l2r_2015_label.ser.gz";
			}

			labels.writeObject(MIRPath.TREC_CDS_2014_DIR + labelFileName);
			data.writeObject(MIRPath.TREC_CDS_2014_DIR + dataFileName);
		}
	}

	public static void train() throws Exception {

		IntegerArray Y = new IntegerArray();
		DenseMatrix X = new DenseMatrix();

		IntegerArray Yt = new IntegerArray();
		DenseMatrix Xt = new DenseMatrix();

		Y.readObject(MIRPath.TREC_CDS_2014_DIR + "l2r_2014_label.ser.gz");
		X.readObject(MIRPath.TREC_CDS_2014_DIR + "l2r_2014_data.ser.gz");

		Yt.readObject(MIRPath.TREC_CDS_2014_DIR + "l2r_2015_label.ser.gz");
		Xt.readObject(MIRPath.TREC_CDS_2014_DIR + "l2r_2015_data.ser.gz");

		for (int i = 0; i < Y.size(); i++) {
			int relevance = Y.get(i);
			int label = 0;
			if (relevance > 0) {
				label = 1;
			}
			Y.set(i, label);
		}

		for (int i = 0; i < Yt.size(); i++) {
			int relevance = Yt.get(i);
			int label = 0;
			if (relevance > 0) {
				label = 1;
			}
			Yt.set(i, label);
		}

		CounterMap<String, Integer> cm = Generics.newCounterMap();

		for (int i = 0; i < Y.size(); i++) {
			int label = Y.get(i);
			cm.incrementCount("Train", label, 1);

			// if (label > 0) {
			// label = 1;
			// }

			Y.set(i, label);
		}

		for (int i = 0; i < Yt.size(); i++) {
			int label = Yt.get(i);
			cm.incrementCount("Test", label, 1);
			// if (label > 0) {
			// label = 1;
			// }

			Yt.set(i, label);
		}

		System.out.println(cm);

		NeuralNetParams param = new NeuralNetParams();
		param.setBatchSize(10);
		param.setLearnRate(0.001);
		param.setRegLambda(0.001);
		param.setThreadSize(2);

		Counter<Integer> labels = Generics.newCounter();
		for (int y : Y) {
			labels.incrementCount(y, 1);
		}

		int input_size = X.colSize();
		int l1_size = 200;
		int l2_size = 50;
		int output_size = labels.size();

		NeuralNet nn = new NeuralNet();
		nn.add(new FullyConnectedLayer(input_size, l1_size));
		nn.add(new BatchNormalization(l1_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		// nn.add(new DropoutLayer(l1_size));
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		nn.add(new BatchNormalization(l2_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));

		nn.prepare();
		nn.init();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), null);
		trainer.train(X, Y, Xt, Yt, 100);
		trainer.finish();

		IntegerArray Yth = nn.classify(Xt);

		// FileUtils.writeStringCollection(KPPath.KYP_DIR +
		// "phrs_test_S_res.txt.gz", lines);
	}

}
