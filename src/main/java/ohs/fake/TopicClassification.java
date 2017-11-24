package ohs.fake;

import java.util.List;

import org.apache.lucene.util.ArrayUtil;

import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.io.TextFileReader;
import ohs.ir.search.app.WordSearcher;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayChecker;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.centroid.CentroidClassifier;
import ohs.ml.eval.Performance;
import ohs.ml.eval.PerformanceEvaluator;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.com.TaskType;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.ml.svm.wrapper.LibLinearTrainer;
import ohs.ml.svm.wrapper.LibLinearWrapper;
import ohs.nlp.ling.types.LDocument;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class TopicClassification {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		TopicClassification c = new TopicClassification();
		c.prepareSvmData();
		c.trainLibLinear();
		// c.trainNN();

		System.out.println("process ends.");
	}

	public void prepareNNData() throws Exception {
		String dir = FNPath.NAVER_DATA_DIR;
		String emdFileName = dir + "emb/glove_ra.ser";
		String vocabFileName = dir + "col/dc/vocab.ser";

		Vocab vocab = DocumentCollection.readVocab(vocabFileName);

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(emdFileName, true);

		WordSearcher ws = new WordSearcher(vocab, E, null);

		Indexer<String> labeIdxer = Generics.newIndexer();

		TextFileReader reader = new TextFileReader(FNPath.NAVER_DATA_DIR + "news_2007.txt");

		List<DenseVector> X = Generics.newLinkedList();
		List<Double> Y = Generics.newArrayList();
		int doc_cnt = 0;

		while (reader.hasNext()) {
			reader.printProgress();

			List<String> ps = StrUtils.split("\t", reader.next());
			ps = StrUtils.unwrap(ps);

			LDocument d = LDocument.newDocument(ps.get(1));
			d.getAttrMap().put("label", ps.get(0));

			DenseVector x = new DenseVector(E.colSize());
			double num_words = 0;
			for (String word : d.getTokenStrings(0)) {
				double doc_freq = vocab.getDocFreq(word);
				double cnt = vocab.getCount(word);
				double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), doc_freq);

				DenseVector e = ws.getVector(word);

				if (e != null) {
					VectorMath.addAfterMultiply(e, tfidf, x);
					num_words++;
				}
			}

			if (num_words < 20) {
				continue;
			}

			x.multiply(1d / num_words);

			int y = labeIdxer.getIndex(d.getAttrMap().get("label"));

			X.add(x);
			Y.add((double) y);

			if (doc_cnt++ > 200000) {
				break;
			}
		}
		reader.printProgress();
		reader.close();

		new DenseMatrix(X).writeObject(FNPath.DATA_DIR + "topic_X.ser.gz");
		new DenseVector(Y).writeObject(FNPath.DATA_DIR + "topic_Y.ser.gz");

		FileUtils.writeStringIndexerAsText(FNPath.DATA_DIR + "topic_idxer.txt", labeIdxer);
	}

	public void prepareSvmData() throws Exception {
		String dir = FNPath.NAVER_DATA_DIR;
		String emdFileName = dir + "emb/glove_ra.ser";
		String vocabFileName = dir + "col/dc/vocab.ser";

		Vocab vocab = DocumentCollection.readVocab(vocabFileName);

		Indexer<String> labeIdxer = Generics.newIndexer();

		TextFileReader reader = new TextFileReader(FNPath.NAVER_DATA_DIR + "news_2007.txt");

		List<SparseVector> X = Generics.newArrayList();
		List<Double> Y = Generics.newArrayList();
		int doc_cnt = 0;

		while (reader.hasNext()) {
			reader.printProgress();

			List<String> ps = StrUtils.split("\t", reader.next());
			ps = StrUtils.unwrap(ps);

			LDocument d = LDocument.newDocument(ps.get(1));
			d.getAttrMap().put("label", ps.get(0));

			Counter<String> c = Generics.newCounter();

			for (String word : d.getTokenStrings(0)) {
				int w = vocab.indexOf(word);
				if (w < 0) {
					continue;
				}
				c.incrementCount(word, 1);
			}

			if (c.totalCount() < 20) {
				continue;
			}

			int y = labeIdxer.getIndex(d.getAttrMap().get("label"));

			X.add(VectorUtils.toSparseVector(c, vocab));

			Y.add((double) y);

			if (doc_cnt++ > 400000) {
				break;
			}
		}
		reader.printProgress();
		reader.close();

		new SparseMatrix(X).writeObject(FNPath.DATA_DIR + "topic_X.ser.gz");
		new DenseVector(Y).writeObject(FNPath.DATA_DIR + "topic_Y.ser.gz");

		FileUtils.writeStringIndexerAsText(FNPath.DATA_DIR + "topic_idxer.txt", labeIdxer);
	}

	public void trainLibLinear() throws Exception {
		SparseMatrix X = new SparseMatrix();
		DenseVector Y = new DenseVector();

		SparseMatrix Xt = new SparseMatrix();
		DenseVector Yt = new DenseVector();

		Indexer<String> labelIdxer = FileUtils.readStringIndexerFromText(FNPath.DATA_DIR + "topic_idxer.txt");

		Vocab vocab = DocumentCollection.readVocab(FNPath.NAVER_DATA_DIR + "col/dc/vocab.ser");

		{

			SparseMatrix DX = new SparseMatrix(FNPath.DATA_DIR + "topic_X.ser.gz");
			DenseVector DY = new DenseVector(FNPath.DATA_DIR + "topic_Y.ser.gz");

			IntegerArray L = new IntegerArray(DY.values());

			IntegerMatrix G = DataSplitter.splitGroupsByLabels(L, new int[] { 500, Integer.MAX_VALUE });

			Yt = new DenseVector(G.get(0).size());
			Y = new DenseVector(G.get(1).size());

			int k1 = 0;
			int k2 = 0;

			for (int i = 0; i < G.size(); i++) {
				IntegerArray Gm = G.get(i);
				ArrayUtils.shuffle(Gm.values());

				for (int j = 0; j < Gm.size(); j++) {
					int loc = Gm.get(j);
					if (i == 0) {
						Xt.add(DX.row(loc));
						Yt.add(k2++, DY.value(loc));
					} else {
						X.add(DX.row(loc));
						Y.add(k1++, DY.value(loc));
					}
				}
			}
		}

		{
			for (SparseVector x : X) {
				x.sortIndexes();
			}

			for (SparseVector x : Xt) {
				x.sortIndexes();
			}

			LibLinearTrainer t = new LibLinearTrainer();
			LibLinearWrapper m = t.train(labelIdxer, vocab, X, Y);
			m.write(FNPath.DATA_DIR + "topic_model.txt");

			DenseVector Yh = new DenseVector(Yt.size());

			for (int i = 0; i < Xt.size(); i++) {
				SparseVector x = Xt.rowAt(i);
				DenseVector yh = m.score(x).toDenseVector(labelIdxer.size());
				Yh.add(i, yh.argMax());
			}

			PerformanceEvaluator pe = new PerformanceEvaluator();
			Performance p = pe.evalute(Yt, Yh, labelIdxer);

			System.out.println(p);

		}
	}

	public void trainNN() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setInputSize(100);
		nnp.setHiddenSize(50);
		nnp.setOutputSize(10);

		nnp.setBatchSize(10);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setGradientClipCutoff(5);
		nnp.setOptimizerType(OptimizerType.ADAM);

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();

		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		Indexer<String> labelIdxer = FileUtils.readStringIndexerFromText(FNPath.DATA_DIR + "topic_idxer.txt");
		Vocab vocab = new Vocab();

		{

			DenseMatrix DX = new DenseMatrix(FNPath.DATA_DIR + "topic_X.ser.gz");
			DenseVector DY = new DenseVector(FNPath.DATA_DIR + "topic_Y.ser.gz");

			IntegerArray L = new IntegerArray(DY.values());

			IntegerMatrix G = DataSplitter.splitGroupsByLabels(DataSplitter.groupByLabels(L),
					new int[] { 500, Integer.MAX_VALUE });

			for (int i = 0; i < G.size(); i++) {
				IntegerArray Gm = G.get(i);

				for (int j = 0; j < Gm.size(); j++) {
					int loc = Gm.get(j);
					if (i == 0) {
						Xt.add(DX.row(loc).toDenseMatrix());
						Yt.add(new DenseVector(new double[] { DY.value(loc) }));
					} else {
						X.add(DX.row(loc).toDenseMatrix());
						Y.add(new DenseVector(new double[] { DY.value(loc) }));
					}
				}
			}
		}

		for (DenseMatrix Xm : X) {
			for (DenseVector xm : Xm) {
				if (!ArrayChecker.isValid(xm.values())) {
					System.out.println();
				}
			}
		}

		int vocab_size = 200;
		int l1_size = 100;
		int l2_size = 25;
		int output_size = labelIdxer.size();

		NeuralNet nn = new NeuralNet(labelIdxer, null, TaskType.CLASSIFICATION);

		nn.add(new FullyConnectedLayer(vocab_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(new ReLU()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		// nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(new ReLU()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.prepareTraining();
		nn.initWeights();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
		trainer.train(X, Y, Xt, Yt, 10000);
		trainer.finish();
	}
}
