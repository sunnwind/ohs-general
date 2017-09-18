package ohs.eden.keyphrase.kmine;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.SolverType;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.DocumentCollectionCreator;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.svm.wrapper.LibLinearTrainer;
import ohs.ml.svm.wrapper.LibLinearWrapper;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * 
 * Determine the number of candidate phrases for an input document
 * 
 * @author ohs
 *
 */
public class KPhraseNumberPredictor {

	public static SparseVector extractFeatures(Vocab featIdxer, MDocument doc) {

		Counter<Integer> c1 = Generics.newCounter();
		Counter<Integer> c2 = Generics.newCounter();

		for (MSentence ms : doc) {
			for (MultiToken mt : ms) {
				for (Token t : mt) {
					String word = t.get(0);
					String pos = t.get(1);

					int f1 = featIdxer.indexOf(word);
					int f2 = featIdxer.indexOf(pos);

					if (f1 != -1) {
						c1.incrementCount(f1, 1);
					}

					if (f2 != -1) {
						c2.incrementCount(f2, 1);
					}

				}
			}
		}

		double len_d = c1.totalCount();
		double norm = 0;

		for (Entry<Integer, Double> e : c1.entrySet()) {
			int f = e.getKey();
			double cnt = e.getValue();
			double tfidf = TermWeighting.tfidf(cnt, featIdxer.getDocCnt(), featIdxer.getDocFreq(f));
			c1.setCount(f, tfidf);
			norm += tfidf * tfidf;
		}

		if (norm > 0) {
			norm = Math.sqrt(norm);
			c1.scale(1f / norm);
		}

		Counter<Integer> c3 = Generics.newCounter();

		if (c1.size() > 0) {
			c3.incrementAll(c1);
			c3.incrementAll(c2);
			c1.incrementCount(featIdxer.indexOf("#len_d"), len_d);
		}

		return new SparseVector(c3);
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		vectorizeData();
		trainModel();

		System.out.println("process ends.");
	}

	public static void trainModel() throws Exception {
		SparseMatrix XD = new SparseMatrix(KPPath.KP_DIR + "ext/X_number.ser.gz");
		IntegerArray YD = new IntegerArray(KPPath.KP_DIR + "ext/Y_number.ser.gz");

		SparseMatrix X = new SparseMatrix();
		IntegerArray Y = new IntegerArray();

		SparseMatrix Xt = new SparseMatrix();
		IntegerArray Yt = new IntegerArray();

		int test_size = 1000;

		{
			IntegerArrayMatrix G = DataSplitter.group(YD);

			for (int i = 0; i < G.size(); i++) {
				IntegerArray locs = G.get(i);
				ArrayUtils.shuffle(locs.values());

				if (locs.size() < test_size) {
					test_size = locs.size() / 2;
				}

				IntegerArray sub1 = locs.subArray(0, test_size);
				IntegerArray sub2 = locs.subArray(test_size, locs.size());

				for (int loc : sub1) {
					Xt.add(XD.get(loc));
					Yt.add(YD.get(loc));
				}

				for (int loc : sub2) {
					X.add(XD.get(loc));
					Y.add(YD.get(loc));
				}
			}

			X.trimToSize();
			Y.trimToSize();
			Xt.trimToSize();
			Yt.trimToSize();

			X.unwrapValues();
			Xt.unwrapValues();
		}

		Indexer<String> labelIdxer = Generics.newIndexer();
		Indexer<String> featIdxer = Generics.newIndexer();

		{
			Set<String> labels = Generics.newTreeSet();
			for (int y : Y) {
				labels.add(y + "");
			}
			labelIdxer = Generics.newIndexer(labels);

			Vocab vocab = DocumentCollection.readVocab(KPPath.KP_DIR + "ext/vocab_num_pred.ser");
			featIdxer = Generics.newIndexer(vocab);
		}

		LibLinearTrainer trainer = new LibLinearTrainer();
		trainer.getParameter().setSolverType(SolverType.L2R_L2LOSS_SVR);
		trainer.getParameter().setEps(0.001);

		LibLinearWrapper llw = trainer.train(labelIdxer, featIdxer, X, Y);

		Model model = llw.getModel();
		Linear.saveModel(new File(KPPath.KP_DIR + "ext/model_num_pred.txt"), model);

		IntegerArray Yh = new IntegerArray(Yt.size());
		double diff = 0;

		for (int i = 0; i < Xt.size(); i++) {
			int y = Yt.get(i);
			SparseVector x = Xt.get(i);

			Feature[] fs = LibLinearWrapper.toFeatures(x, model.getNrFeature(), model.getBias());
			double yh = Linear.predict(model, fs);
			yh = Math.round(yh);
			Yh.add((int) yh);
			System.out.printf("(%d, %d, %d)\n", y, (int) yh, (int) (y - yh));
			diff += Math.abs(y - yh);
		}

		diff /= Yt.size();

		System.out.println(diff);

	}

	public static void vectorizeData() throws Exception {
		List<String> ins = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");

		List<SparseVector> X = Generics.newArrayList(ins.size());
		List<Integer> Y = Generics.newArrayList(ins.size());

		Counter<String> wordCnts = Generics.newCounter();
		Counter<String> docFreqs = Generics.newCounter();
		Counter<String> posCnts = Generics.newCounter();

		for (String line : ins) {
			String[] ps = line.split("\t");
			MDocument md = MDocument.newDocument(ps[2]);

			for (MSentence ms : md) {
				Counter<String> c = Generics.newCounter();
				for (Token t : ms.getTokens()) {
					String word = t.get(0);
					String pos = t.get(1);
					c.incrementCount(word, 1);
					posCnts.incrementCount(pos, 1);
				}
				wordCnts.incrementAll(c);

				for (String word : c.keySet()) {
					docFreqs.incrementCount(word, 1);
				}
			}
		}

		System.out.println(wordCnts.toString());
		System.out.println(posCnts.toString());

		Vocab featIdxer = new Vocab();
		featIdxer.addAll(wordCnts.keySet());
		featIdxer.addAll(posCnts.keySet());

		featIdxer.add("#len_d");
		featIdxer.add("#avg_idf");
		featIdxer.add("#noun_ratio");
		featIdxer.add("#verb_ratio");

		featIdxer.setDocCnt(ins.size());

		featIdxer.setWordCnts(new IntegerArray(new int[featIdxer.size()]));
		featIdxer.setDocFreqs(new IntegerArray(new int[featIdxer.size()]));

		{
			IntegerArray cnts = featIdxer.getCounts();
			IntegerArray freqs = featIdxer.getDocFreqs();

			for (String word : wordCnts.keySet()) {
				int w = featIdxer.indexOf(word);
				double cnt = wordCnts.getCount(word);
				double doc_freq = docFreqs.getCount(word);
				cnts.set(w, (int) cnt);
				freqs.set(w, (int) doc_freq);
			}
		}

		{
			IntegerArray cnts = featIdxer.getCounts();

			for (String pos : posCnts.keySet()) {
				int f = featIdxer.indexOf(pos);
				double cnt = posCnts.getCount(pos);
				cnts.set(f, (int) cnt);
			}
		}

		for (String line : ins) {
			String[] ps = line.split("\t");
			MDocument md1 = MDocument.newDocument(ps[1]);
			MDocument md2 = MDocument.newDocument(ps[2]);
			SparseVector x = extractFeatures(featIdxer, md2);

			if (x.size() == 0) {
				continue;
			}

			int phrs_size = md1.size();

			Y.add(phrs_size);
			X.add(x);
		}

		new SparseMatrix(X).writeObject(KPPath.KP_DIR + "ext/X_number.ser.gz");
		new IntegerArray(Y).writeObject(KPPath.KP_DIR + "ext/Y_number.ser.gz");

		DocumentCollectionCreator.writeVocab(featIdxer, KPPath.KP_DIR + "ext/vocab_num_pred.ser");
	}

	public String delim = "  ";

	private Model model;

	private Vocab vocab;

	public KPhraseNumberPredictor(Vocab vocab, Model model) {
		this.vocab = vocab;
		this.model = model;
	}

	private DenseVector extractFeatures(RandomAccessDenseMatrix E, Vocab vocab, String kwd) throws Exception {
		IntegerArray Q = vocab.indexesOf(kwd.split(" "));

		StringBuffer sb = new StringBuffer();

		for (int w : Q) {
			String word = vocab.getObject(w);
			sb.append(word + " ");
		}

		String s = sb.toString().trim();

		int unseen_cnt = 0;

		for (int w : Q) {
			if (w < 0) {
				unseen_cnt++;
			}
		}

		if (unseen_cnt > 0) {
			return null;
		}

		DenseMatrix E2 = E.rowsAsMatrix(Q.values()).copy();

		Map<Integer, DenseVector> set = Generics.newHashMap();

		for (int i = 0; i < Q.size(); i++) {
			set.put(Q.get(i), E2.get(i));
		}

		DenseVector ne1 = new DenseVector(E.colSize());
		DenseVector ne2 = new DenseVector(E.colSize());

		for (int i = 0; i < E2.rowSize(); i++) {
			DenseVector e = E2.row(i);
			VectorMath.add(e, ne1);

			if (i == 0 || i < E2.rowSize() - 1) {
				VectorMath.add(e, ne2);
			}
		}

		ne1.multiply(1f / Q.size());
		ne2.multiply(1f / 2);

		DenseMatrix ret = new DenseMatrix(new DenseVector[] { ne1, ne1 });
		return ret.toDenseVector();
	}

	public IntegerArray predict(List<SparseVector> X) {
		IntegerArray ret = new IntegerArray(X.size());
		for (SparseVector x : X) {
			ret.add(predict(x));
		}
		return ret;
	}

	public int predict(MDocument doc) {
		SparseVector x = extractFeatures(vocab, doc);
		return predict(x);
	}

	public int predict(SparseVector x) {
		Feature[] fs = LibLinearWrapper.toFeatures(x, model.getNrFeature(), model.getBias());
		double yh = Linear.predict(model, fs);
		yh = Math.round(yh);
		return (int) yh;
	}

}
