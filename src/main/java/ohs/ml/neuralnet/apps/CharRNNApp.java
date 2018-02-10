package ohs.ml.neuralnet.apps;

import java.util.List;
import java.util.Set;

import ohs.corpus.type.EnglishTokenizer;
import ohs.io.FileUtils;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.com.SentenceGenerator;
import ohs.ml.neuralnet.com.TaskType;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.RnnLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.types.generic.Indexer;
import ohs.types.generic.Vocab;
import ohs.types.generic.Vocab.SYM;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class CharRNNApp {

	public static void run() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(25);
		nnp.setIsFullSequenceBatch(false);
		nnp.setIsRandomBatch(false);
		nnp.setGradientAccumulatorResetSize(100);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setL2WeightDecay(0.9999);
		nnp.setGradientDecay(1);
		nnp.setK2(1);
		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(5);

		DenseTensor X = new DenseTensor();
		DenseTensor Y = new DenseTensor();
		DenseTensor Xt = new DenseTensor();
		DenseTensor Yt = new DenseTensor();

		List<String> lines = Generics.newLinkedList();

		EnglishTokenizer et = new EnglishTokenizer();

		for (String line : FileUtils.readFromText("../../data/ml_data/shakespeare_input.txt").split("\n\n")) {
			lines.add(StrUtils.join(" ", et.tokenize(line)));
		}

		lines = Generics.newArrayList(lines);

		lines = lines.subList(0, 1000);

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
			}

			for (String s : set) {
				vocab.add(s);
			}
		}

		for (int i = 0; i < lines.size(); i++) {
			String s = lines.get(i);

			IntegerArray t = new IntegerArray(s.length() + 2);
			t.add(vocab.indexOf(SYM.START.getText()));

			for (int j = 0; j < s.length(); j++) {
				int w = vocab.indexOf(s.charAt(j) + "", 0);
				t.add(w);
			}

			t.add(vocab.indexOf(SYM.END.getText()));

			DenseMatrix Xm = new DenseMatrix(t.size() - 1, 1);
			DenseMatrix Ym = new DenseMatrix(t.size() - 1, 1);

			for (int j = 0; j < t.size() - 1; j++) {
				Xm.add(j, 0, t.get(j));
				Ym.add(j, 0, t.get(j + 1));
			}

			if (i < train_size) {
				X.add(Xm);
				Y.add(Ym);
			} else {
				Xt.add(Xm);
				Yt.add(Ym);
			}
		}

		Indexer<String> labelIdxer = Generics.newIndexer(vocab.getObjects());

		int voc_size = vocab.size();
		int emb_size = 100;
		int l1_size = 100;
		int label_size = labelIdxer.size();
		int type = 2;
		int k1 = nnp.getK1();
		int k2 = nnp.getK2();

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.TOKEN_SEQ_LABELING);

		String modelFileName = "../../data/ml_data/char-rnn.ser";

		if (FileUtils.exists(modelFileName)) {
			nn = new NeuralNet(modelFileName);
			nn.createGradientHolders();
		} else {

			// EmbeddingLayer l = new EmbeddingLayer(voc_size, emb_size, true);
			// l.setOutputWordIndexes(false);
			// nn.add(l);

			nn.add(new EmbeddingLayer(voc_size, emb_size, true, 0));
			nn.add(new DropoutLayer());
			// nn.add(new BidirectionalRecurrentLayer(Type.LSTM, emb_size, l1_size,
			// bptt, new ReLU()));
			// nn.add(new LstmLayer(emb_size, l1_size));
			nn.add(new RnnLayer(emb_size, l1_size, k1, k2, new ReLU()));
			// nn.add(new DropoutLayer());
			// nn.add(new BatchNormalizationLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, label_size));
			nn.add(new SoftmaxLayer(label_size));
			nn.createGradientHolders();
			nn.initWeights(new ParameterInitializer());
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
					DenseTensor Ym = new DenseTensor();

					Xm.ensureCapacity(r_size);
					Ym.ensureCapacity(r_size);

					for (int k = r[0]; k < r[1]; k++) {
						int loc = locs.get(k);
						Xm.add(X.get(loc));
						Ym.add(Y.get(loc));
					}

					trainer.train(Xm, Ym, null, null, 1);
				}
			}

			if (u % 10 == 0) {
				for (int j = 0; j < 10; j++) {
					String s = sg.generate(100);
					System.out.println(s);
				}
			}

		}

		trainer.finish();
	}
}
