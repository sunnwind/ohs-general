package ohs.ml.neuralnet.com;

import java.util.List;

import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.RecurrentLayer;
import ohs.types.generic.Vocab;
import ohs.types.generic.Vocab.SYM;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class SentenceGenerator {

	private NeuralNet nn;

	private Vocab vocab;

	public SentenceGenerator(NeuralNet nn, Vocab vocab) {
		super();
		this.nn = nn;
		this.vocab = vocab;
	}

	public String generate(int max_words) {
		int w_unk = vocab.indexOf(SYM.UNK.getText());
		int w_start = vocab.indexOf(SYM.START.getText());
		int w_end = vocab.indexOf(SYM.END.getText());

		List<Integer> s1 = Generics.newArrayList(max_words);
		s1.add(w_start);

		List<String> s2 = Generics.newArrayList(max_words);
		s2.add(vocab.getObject(w_start));

		while (s1.get(s1.size() - 1) != w_end) {
			if (s1.size() - 1 == max_words) {
				s1.add(w_end);
				break;
			}

			DenseMatrix X = new DenseMatrix();

			for (int i = 0; i < s1.size(); i++) {
				X.add(new DenseVector(new double[] { s1.get(i) }));
			}

			DenseTensor Yh = (DenseTensor) nn.forward(X.toDenseTensor());
			DenseVector yh = Yh.get(0).row(Yh.get(0).size() - 1);

			VectorMath.cumulateAfterNormalize(yh, yh);

			int w = w_unk;
			int cnt = 0;

			do {
				if (cnt++ > 10) {
					SparseVector sv = yh.toSparseVector();
					w = sv.argMax();
					break;
				}
				w = ArrayMath.sample(yh.values());
			} while (w == w_unk || w == w_start);

			s1.add(w);
			s2.add(vocab.getObject(w));
		}

		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < s1.size(); i++) {
			sb.append(vocab.getObject(s1.get(i)));
		}

		return sb.toString();
	}

}