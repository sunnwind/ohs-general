package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.Set;

import ohs.corpus.type.EnglishTokenizer;
import ohs.io.FileUtils;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.ConvolutionalLayer;
import ohs.ml.neuralnet.layer.DiscreteFeatureEmbeddingLayer;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.MaxPoolingLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.RecurrentLayer.Type;
import ohs.ml.neuralnet.layer.RnnLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.ml.neuralnet.nonlinearity.Tanh;
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

public class Apps {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// testMNIST();
		// testCharRNN();
		// testNER();

		testSentenceClassification();

		System.out.println("process ends.");
	}

	public static void testCharRNN() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(5);
		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setGradientAccumulatorResetSize(100);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setTruncatedBackPropagationThroughTime(1);
		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(10);

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();
		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		List<String> lines = Generics.newLinkedList();

		// for (String line :
		// FileUtils.readFromText("../../data/ml_data/love_input.txt").split("\n\n")) {
		// line = StrUtils.normalizeSpaces(line);
		// if (line.length() > 0) {
		// lines.add(line.toLowerCase());
		// }
		// }

		EnglishTokenizer et = new EnglishTokenizer();

		for (String line : FileUtils.readFromText("../../data/ml_data/shakespeare_input.txt").split("\n\n")) {
			lines.add(StrUtils.join(" ", et.tokenize(line)));
		}

		lines = Generics.newArrayList(lines);

		int test_size = 1000;
		// int train_size = lines.size() - test_size;
		int train_size = 10000;

		Vocab vocab = new Vocab();
		vocab.add(SYM.UNK.getText());
		vocab.add(SYM.START.getText());
		vocab.add(SYM.END.getText());

		{
			Set<String> set = Generics.newTreeSet();

			for (int i = 0; i < lines.size() && i < train_size; i++) {
				String s = lines.get(i);
				for (int j = 0; j < s.length(); j++) {
					set.add(s.charAt(j) + "");
				}

				// for (String word : StrUtils.split(s)) {
				// vocab.add(word);
				// }
			}

			for (String s : set) {
				vocab.add(s);
			}
		}

		for (int i = 0; i < lines.size(); i++) {
			String s = lines.get(i);

			IntegerArray t = new IntegerArray(s.length());
			t.add(vocab.indexOf(SYM.START.getText()));

			for (int j = 0; j < s.length(); j++) {
				t.add(vocab.indexOf(s.charAt(j) + "", 0));
			}

			t.add(vocab.indexOf(SYM.END.getText()));

			// IntegerArray t = new IntegerArray(s.length());
			// t.add(vocab.indexOf(SYM.START.getText()));
			//
			// for (String word : StrUtils.split(s)) {
			// t.add(vocab.indexOf(word, 0));
			// }
			// t.add(vocab.indexOf(SYM.END.getText()));

			DenseVector x = new DenseVector(t.size() - 1);
			DenseVector y = new DenseVector(t.size() - 1);

			for (int j = 0; j < t.size() - 1; j++) {
				x.add(j, t.get(j));
				y.add(j, t.get(j + 1));
			}

			if (i < train_size) {
				X.add(x.toDenseMatrix());
				Y.add(y);
			} else {
				Xt.add(x.toDenseMatrix());
				Yt.add(y);
			}
		}

		Indexer<String> labelIdxer = Generics.newIndexer(vocab.getObjects());

		int voc_size = vocab.size();
		int emb_size = 100;
		int l1_size = 100;
		int label_size = labelIdxer.size();
		int type = 2;
		int bptt = nnp.getBPTTSize();

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_LABELING);

		String modelFileName = "../../data/ml_data/char-rnn.ser";

		if (FileUtils.exists(modelFileName)) {
			nn = new NeuralNet(modelFileName);
			nn.prepare();
		} else {
			nn.add(new EmbeddingLayer(voc_size, emb_size, true));
			nn.add(new DropoutLayer());
			// nn.add(new BidirectionalRecurrentLayer(Type.LSTM, emb_size, l1_size,
			// bptt, new ReLU()));
			// nn.add(new LstmLayer(emb_size, l1_size, new ReLU()));
			nn.add(new RnnLayer(emb_size, l1_size, bptt, new ReLU()));
			// nn.add(new BatchNormalizationLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, label_size));
			nn.add(new SoftmaxLayer(label_size));
			nn.prepare();
			nn.init();
		}

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);

		IntegerArray locs = new IntegerArray(ArrayUtils.range(X.size()));

		int group_size = 1000;
		int[][] rs = BatchUtils.getBatchRanges(X.size(), group_size);

		int max_iters = 1000;

		SentenceGenerator sg = new SentenceGenerator(nn, vocab);

		for (int u = 0; u < max_iters; u++) {
			for (int i = 0; i < rs.length; i++) {
				System.out.printf("iters [%d/%d], batches [%d/%d]\n", u + 1, max_iters, i + 1, rs.length);
				for (int j = 0; j < rs.length; j++) {
					int[] r = rs[j];
					int r_size = r[1] - r[0];
					DenseTensor Xm = new DenseTensor();
					DenseMatrix Ym = new DenseMatrix();

					Xm.ensureCapacity(r_size);
					Ym.ensureCapacity(r_size);

					for (int k = r[0]; k < r[1]; k++) {
						int loc = locs.get(k);
						Xm.add(X.get(loc));
						Ym.add(Y.get(loc));
					}

					trainer.train(Xm, Ym, null, null, 1);
				}

				if (u % 10 == 0) {
					for (int j = 0; j < 10; j++) {
						String s = sg.generate(100);
						System.out.println(s);
					}
				}
			}
		}

		trainer.finish();
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

	public static void testNER() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(5);
		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setGradientAccumulatorResetSize(1000);
		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(0.9);
		nnp.setLearnRateDecaySize(100);
		nnp.setWeightDecay(1);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setTruncatedBackPropagationThroughTime(1);
		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(5);

		String trainFileName = "../../data/ml_data/conll2003.bio2/train.dat";
		String testFileName = "../../data/ml_data/conll2003.bio2/test.dat";

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			l.add("pos");
			l.add("chunk");
			l.add("ner");
			MToken.INDEXER = l;
		}

		MDocument trainData = MDocument.newDocument(FileUtils.readFromText(trainFileName));
		MDocument testData = MDocument.newDocument(FileUtils.readFromText(testFileName));

		Indexer<String> labelIdxer = Generics.newIndexer();
		Vocab vocab = new Vocab();
		vocab.add(Vocab.SYM.UNK.getText());

		int unk = 0;

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();

		X.ensureCapacity(trainData.size());
		Y.ensureCapacity(trainData.size());

		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		Xt.ensureCapacity(trainData.size());
		Xt.ensureCapacity(trainData.size());

		WordFeatureExtractor ext = new WordFeatureExtractor();

		{
			MDocument D = new MDocument();
			D.addAll(trainData);
			D.addAll(testData);

			Set<String> labels = Generics.newTreeSet();

			for (MToken t : D.getTokens()) {
				String word = t.getString(0);
				String ner = t.getString(3);
				String[] ps = ner.split("-");
				if (ps.length == 2) {
					ner = String.format("%s-%s", ps[1], ps[0]);
				}
				t.set(3, ner);
				ext.extract(t);

				labels.add(ner);
			}

			labels.remove("O");
			labelIdxer.addAll(labels);
			labelIdxer.add("O");

			for (int i = 0; i < D.size(); i++) {
				MSentence s = D.get(i);

				DenseMatrix Xm = new DenseMatrix();
				DenseVector Ym = new DenseVector(s.size());

				Xm.ensureCapacity(s.size());

				for (int j = 0; j < s.size(); j++) {
					MToken t = s.get(j);

					IntegerArray f = new IntegerArray(1 + ext.getFeatureIndexer().size());

					if (i < trainData.size()) {
						f.add(vocab.getIndex(t.getString(4)));
						f.addAll((IntegerArray) t.get(5));

						Xm.add(new DenseVector(f.values()));
						Ym.add(j, labelIdxer.getIndex(t.getString(3)));
					} else {
						f.add(vocab.indexOf(t.getString(4), 0));
						f.addAll((IntegerArray) t.get(5));

						Xm.add(new DenseVector(f.values()));
						Ym.add(j, labelIdxer.getIndex(t.getString(3)));
					}
				}

				if (i < trainData.size()) {
					X.add(Xm);
					Y.add(Ym);
				} else {
					Xt.add(Xm);
					Yt.add(Ym);
				}
			}
		}

		X.trimToSize();
		Y.trimToSize();

		Xt.trimToSize();
		Yt.trimToSize();

		System.out.println(vocab.info());
		System.out.println(labelIdxer.info());
		System.out.println(X.sizeOfEntries());

		int voc_size = vocab.size();
		int word_emb_size = 50;
		int feat_emb_size = 10;
		int l1_size = 100;
		int label_size = labelIdxer.size();
		int type = 2;
		int bptt_size = nnp.getBPTTSize();

		String modelFileName = "../../data/ml_data/ner_nn.ser";

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_LABELING);

		boolean read_ner_model = false;

		if (read_ner_model && FileUtils.exists(modelFileName)) {
			nn = new NeuralNet(modelFileName);
			nn.prepare();
		} else {
			nn.add(new EmbeddingLayer(voc_size, word_emb_size, true));

			DiscreteFeatureEmbeddingLayer l = new DiscreteFeatureEmbeddingLayer(ext.getFeatureIndexer().size(),
					feat_emb_size, word_emb_size, true);
			nn.add(l);
			// nn.add(new DropoutLayer());
			// nn.add(new RnnLayer(l.getOutputSize(), l1_size, bptt_size, new ReLU()));
			// nn.add(new LstmLayer(l.getOutputSize(), l1_size));
			nn.add(new BidirectionalRecurrentLayer(Type.LSTM, l.getOutputSize(), l1_size, bptt_size, new ReLU()));
			// nn.add(new BatchNormalizationLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, label_size));
			nn.add(new SoftmaxLayer(label_size));
			nn.prepare();
			nn.init();
		}

		// nn.writeObject(modelFileName);

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);

		int max_iters = 1000;
		boolean use_batches = true;

		if (use_batches) {
			IntegerArray locs = new IntegerArray(ArrayUtils.range(X.size()));
			int group_size = 10000;
			int[][] rs = BatchUtils.getBatchRanges(X.size(), group_size);

			for (int u = 0; u < max_iters; u++) {
				ArrayUtils.shuffle(locs.values());

				for (int i = 0; i < rs.length; i++) {
					System.out.printf("iters [%d/%d], batches [%d/%d]\n", u + 1, max_iters, i + 1, rs.length);
					for (int j = 0; j < rs.length; j++) {
						int[] r = rs[j];
						int r_size = r[1] - r[0];
						DenseTensor Xm = new DenseTensor();
						DenseMatrix Ym = new DenseMatrix();

						Xm.ensureCapacity(r_size);
						Ym.ensureCapacity(r_size);

						for (int k = r[0]; k < r[1]; k++) {
							int loc = locs.get(k);
							Xm.add(X.get(loc));
							Ym.add(Y.get(loc));
						}

						trainer.train(Xm, Ym, Xt, Yt, 1);
					}
				}
			}
		} else {
			trainer.train(X, Y, Xt, Yt, max_iters);
		}

		trainer.finish();

		nn.writeObject(modelFileName);
	}

	public static void testSentenceClassification() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(5);
		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setGradientAccumulatorResetSize(1000);
		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(0.9);
		nnp.setLearnRateDecaySize(100);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setTruncatedBackPropagationThroughTime(1);
		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(5);

		Indexer<String> labelIdxer = Generics.newIndexer();
		labelIdxer.add("pos");
		labelIdxer.add("neg");

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			MToken.INDEXER = l;
		}

		MDocument posData = new MDocument();
		MDocument negData = new MDocument();

		for (String line : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.pos")) {
			line = StrUtils.normalizeSpaces(line);
			MSentence s = MSentence.newSentence(line.replace(" ", "\n"));
			posData.add(s);

			s.getAttrMap().put("por", "pos");
		}

		for (String line : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.neg")) {
			line = StrUtils.normalizeSpaces(line);
			MSentence s = MSentence.newSentence(line.replace(" ", "\n"));
			negData.add(s);

			s.getAttrMap().put("por", "neg");
		}

		Vocab vocab = new Vocab();
		vocab.add(SYM.UNK.getText());

		for (MToken t : posData.getTokens()) {
			vocab.add(t.getString(0));
		}

		MDocument data = new MDocument();
		data.addAll(posData);
		data.addAll(negData);

		IntegerMatrix M = new IntegerMatrix();
		IntegerArray N = new IntegerArray();

		for (int i = 0; i < data.size(); i++) {
			MSentence s = data.get(i);
			String por = s.getAttrMap().get("por");
			int label = labelIdxer.indexOf(por);

			M.add(label, i);
			N.add(label);
		}

		IntegerMatrix T = DataSplitter.splitGroups(M, new int[] { 5000, 500 });

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();
		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		for (int i = 0; i < T.size(); i++) {
			IntegerArray L = T.get(i);
			for (int loc : L) {
				MSentence s = data.get(loc);
				String por = s.getAttrMap().get("por");
				int label = labelIdxer.indexOf(por);

				List<Integer> ws = vocab.indexesOf(s.getTokenStrings(0), 0);

				DenseMatrix Xm = new DenseMatrix(ws.size(), 1);
				for (int j = 0; j < ws.size(); j++) {
					Xm.add(j, 0, ws.get(j));
				}

				if (i == 0) {
					X.add(Xm);
					Y.add(new DenseVector(new int[] { label }));
				} else {
					Xt.add(Xm);
					Yt.add(new DenseVector(new int[] { label }));
				}
			}
		}

		X.trimToSize();
		Y.trimToSize();

		Xt.trimToSize();
		Yt.trimToSize();

		int size1 = 0;
		int size2 = 0;
		int max_len = 0;
		int min_len = Integer.MAX_VALUE;

		int pad_label = vocab.getIndex("<PAD>");

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

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_CLASSIFICATION);

		if (type == 0) {
			int num_filters = 100;
			int[] window_sizes = new int[] { 2, 3 };

			nn.add(new EmbeddingLayer(vocab_size, emb_size, true));

//			nn.add(new MultiWindowConvolutionalLayer(emb_size, window_sizes, num_filters));
			 nn.add(new ConvolutionalLayer(emb_size, window_sizes[0], num_filters));
			nn.add(new NonlinearityLayer(new ReLU()));
			nn.add(new MaxPoolingLayer(num_filters));
			nn.add(new DropoutLayer());
			nn.add(new FullyConnectedLayer(num_filters, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
			trainer.train(X, Y, Xt, Yt, 10000);
			trainer.finish();
		}

	}
}