package ohs.fake;

import java.io.File;
import java.util.List;
import java.util.Set;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.EnglishTokenizer;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.io.TextFileWriter;
import ohs.ir.search.app.WordSearcher;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.ml.neuralnet.com.DataReader;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetMultiRunner;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.com.SentenceGenerator;
import ohs.ml.neuralnet.com.TaskType;
import ohs.ml.neuralnet.com.WordFeatureExtractor;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.DiscreteFeatureEmbeddingLayer;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.MultiWindowConvolutionalLayer;
import ohs.ml.neuralnet.layer.MultiWindowMaxPoolingLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.RecurrentLayer.Type;
import ohs.ml.neuralnet.layer.RnnLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.nlp.ling.types.MCollection;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.generic.Vocab.SYM;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class FakeApps {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// testMNIST();
		// testCharRNN();
		// testNER();s

		// testSentenceClassification();
		runM1();

		System.out.println("process ends.");
	}

	public static void runM1() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();

		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(0.9);
		nnp.setLearnRateDecaySize(100);
		nnp.setWeightDecayL2(1);
		nnp.setGradientDecay(1);
		nnp.setRegLambda(0.001);
		nnp.setGradientClipCutoff(5);

		nnp.setThreadSize(5);
		nnp.setBatchSize(1);
		nnp.setGradientAccumulatorResetSize(1000);
		nnp.setBPTT(1);

		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setUseAverageGradients(false);
		nnp.setUseHardGradientClipping(false);

		nnp.setOptimizerType(OptimizerType.ADAM);

		Indexer<String> labelIdxer = Generics.newIndexer();
		labelIdxer.add("non-fake");
		labelIdxer.add("fake");

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			l.add("pos");
			MToken.INDEXER = l;
		}

		String[] fileNames = { "M1_train_pos.txt", "M1_test_pos" };

		File inFile = new File(FNPath.DATA_DIR + "data", fileNames[0]);

		MCollection train = new MCollection();
		MCollection test = new MCollection();

		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data")) {
			if (file.getName().startsWith("M1_train_pos")) {
				for (String line : FileUtils.readLinesFromText(inFile)) {
					List<String> ps = StrUtils.split("\t", line);
					ps = StrUtils.unwrap(ps);

					{
						String label = ps.get(1);
						label = label.equals("0") ? "non-fake" : "fake";
						MDocument d = MDocument.newDocument(ps.get(2));
						d.getAttrMap().put("label", label);

						if (ps.get(3).length() > 0) {
							d.getAttrMap().put("cor_title", ps.get(3));
						}
						train.add(d);
					}
				}
			} else if (file.getName().startsWith("M1_test_pos")) {
				for (String line : FileUtils.readLinesFromText(inFile)) {
					List<String> ps = StrUtils.split("\t", line);
					ps = StrUtils.unwrap(ps);
					MDocument d = MDocument.newDocument(ps.get(2));
					d.getAttrMap().put("id", ps.get(0));
					test.add(d);
				}
			}
		}

		DataGenerator dg = new DataGenerator(train);
		MCollection extData = dg.generate();

		train.addAll(extData);

		System.out.printf("train size:\t%d\n", train.size());
		System.out.printf("test size:\t%d\n", test.size());
		System.out.println();

		Vocab vocab = new Vocab();
		vocab.add(SYM.UNK.getText());

		IntegerMatrix M = new IntegerMatrix();
		IntegerArray N = new IntegerArray();

		for (int i = 0; i < train.size(); i++) {
			MDocument d = train.get(i);
			String label = d.getAttrMap().get("label");
			int l = labelIdxer.indexOf(label);

			M.add(l, i);
			N.add(l);
		}

		// IntegerMatrix T = DataSplitter.splitGroups(M, new int[] { 150, 25 });

		IntegerMatrix T = DataSplitter.splitGroups(M, new double[] { 0.9, 0.1 });

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();

		DenseTensor Xv = new DenseTensor();
		DenseMatrix Yv = new DenseMatrix();

		DenseTensor Xt = new DenseTensor();

		for (int loc : T.get(0)) {
			MDocument d = train.get(loc);
			for (String word : d.getTokens().getTokenStrings(0)) {
				vocab.add(word);
			}
		}

		for (int i = 0; i < T.size(); i++) {
			IntegerArray L = T.get(i);
			for (int loc : L) {
				MDocument d = train.get(loc);
				String label = d.getAttrMap().get("label");
				int lb = labelIdxer.indexOf(label);

				List<Integer> ws = vocab.indexesOf(d.getTokens().getTokenStrings(0), 0);

				DenseMatrix Xm = new DenseMatrix(ws.size(), 1);
				for (int j = 0; j < ws.size(); j++) {
					Xm.add(j, 0, ws.get(j));
				}

				if (i == 0) {
					X.add(Xm);
					Y.add(new DenseVector(new int[] { lb }));
				} else {
					Xv.add(Xm);
					Yv.add(new DenseVector(new int[] { lb }));
				}
			}
		}

		for (MDocument d : test) {
			String label = d.getAttrMap().get("label");

			List<Integer> ws = vocab.indexesOf(d.getTokens().getTokenStrings(0), 0);

			DenseMatrix Xm = new DenseMatrix(ws.size(), 1);

			for (int j = 0; j < ws.size(); j++) {
				Xm.add(j, 0, ws.get(j));
			}

			Xt.add(Xm);
		}

		X.trimToSize();
		Y.trimToSize();

		Xv.trimToSize();
		Yv.trimToSize();

		int size1 = 0;
		int size2 = 0;
		int max_len = 0;
		int min_len = Integer.MAX_VALUE;

		System.out.printf("data size: [%d -> %d]\n", size1, size2);
		System.out.printf("max len: [%d]\n", max_len);
		System.out.printf("min len: [%d]\n", min_len);

		int vocab_size = vocab.size();
		int emb_size = 200;
		int l1_size = 100;

		int l2_size = 20;
		int output_size = 2;
		int type = 0;

		System.out.println(vocab.info());

		DenseMatrix E = new DenseMatrix();

		{
			String dir = FNPath.NAVER_DATA_DIR;
			String emdFileName = dir + "emb/glove_ra.ser";
			String vocabFileName = dir + "col/dc/vocab.ser";

			Vocab v = DocumentCollection.readVocab(vocabFileName);

			WordSearcher ws = new WordSearcher(v, new RandomAccessDenseMatrix(emdFileName, false), null);

			List<DenseVector> es = Generics.newArrayList(vocab_size);

			for (int i = 0; i < vocab.size(); i++) {
				String word = vocab.getObject(i);
				DenseVector e = ws.getVector(word);

				if (e == null) {
					e = new DenseVector(ws.getEmbeddingMatrix().colSize());
				} else {
					es.add(e);
				}
			}

			E = new DenseMatrix(es);

			// E.add(1d / VectorMath.normL2(E));

			emb_size = E.colSize();
		}

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_CLASSIFICATION);

		if (type == 0) {
			int num_filters = 100;
			int[] window_sizes = new int[] { 2, 3, 5 };

			nn.add(new EmbeddingLayer(vocab_size, emb_size, true));

			nn.add(new MultiWindowConvolutionalLayer(emb_size, window_sizes, num_filters));
			// nn.add(new ConvolutionalLayer(emb_size, 4, num_filters));
			nn.add(new NonlinearityLayer(new ReLU()));
			// nn.add(new MaxPoolingLayer(num_filters));
			nn.add(new MultiWindowMaxPoolingLayer(num_filters, window_sizes));
			nn.add(new DropoutLayer());
			nn.add(new FullyConnectedLayer(num_filters * window_sizes.length, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
			trainer.train(X, Y, Xv, Yv, 1);

			trainer.finish();
		}

		{
			DenseTensor Yt = (DenseTensor) nn.forward(Xt);

			TextFileWriter writer = new TextFileWriter(FNPath.DATA_DIR + "M1_test_out.txt");

			for (int i = 0; i < Yt.size(); i++) {
				DenseVector yh = Yt.row(i).row(0);
				MDocument d = test.get(i);
				String id = d.getAttrMap().get("id");
				writer.write(String.format("%s\t%f\n", id, yh.value(1)));
			}
			writer.close();
		}

	}

	public static void testMNIST() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setInputSize(100);
		nnp.setHiddenSize(50);
		nnp.setOutputSize(10);
		nnp.setBatchSize(50);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setGradientClipCutoff(20);
		nnp.setOptimizerType(OptimizerType.ADAM);

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();

		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		Indexer<String> labelIdxer = Generics.newIndexer();
		Vocab vocab = new Vocab();

		{

			Pair<SparseMatrix, IntegerArray> train = DataReader.readFromSvmFormat("../../data/ml_data/mnist.txt");
			Pair<SparseMatrix, IntegerArray> test = DataReader.readFromSvmFormat("../../data/ml_data/mnist.t.txt");

			DenseMatrix _X = train.getFirst().toDenseMatrix();
			IntegerArray _Y = train.getSecond();

			DenseMatrix _Xt = test.getFirst().toDenseMatrix(test.getFirst().rowSize(), _X.colSize());
			IntegerArray _Yt = test.getSecond();

			Set<String> labels = Generics.newTreeSet();
			for (int y : _Y) {
				labels.add(y + "");
			}

			labelIdxer = Generics.newIndexer(labels);

			for (int i = 0; i < _X.colSize(); i++) {
				vocab.add(i + "");
			}

			X.ensureCapacity(_X.rowSize());
			Y = new DenseMatrix(_X.rowSize(), 1);

			for (int i = 0; i < _X.rowSize(); i++) {
				X.add(_X.row(i).toDenseMatrix());
				Y.add(i, 0, _Y.get(i));
			}

			Xt.ensureCapacity(_Xt.rowSize());
			Yt = new DenseMatrix(_Xt.rowSize(), 1);

			for (int i = 0; i < _Xt.rowSize(); i++) {
				Xt.add(_Xt.row(i).toDenseMatrix());
				Yt.add(i, 0, _Yt.get(i));
			}
		}

		int vocab_size = vocab.size();
		int l1_size = 100;
		int l2_size = 25;
		int output_size = labelIdxer.size();

		NeuralNet nn = new NeuralNet(labelIdxer, null, TaskType.CLASSIFICATION);

		nn.add(new FullyConnectedLayer(vocab_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		// nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.prepare();
		nn.init();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
		trainer.train(X, Y, Xt, Yt, 10000);
		trainer.finish();
	}

}